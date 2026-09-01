// Cloudflare Worker: proxies food-description -> AI nutrition estimate
// requests through a two-step pipeline, so no API key reaches the browser.
// Step 1: DeepL translates Hebrew -> English — a dedicated MT engine, not an
// LLM, chosen after Gemini and Groq's own chat models both proved unreliable
// at processing short Hebrew input directly (see git history for the full
// investigation). MyMemory was tried first but its rate limiting is IP-based
// and Cloudflare Workers share an outbound IP pool across many customers, so
// it got exhausted by traffic outside our control, not just our own usage.
// DeepL is key-authenticated, so our quota is actually ours. Step 2: Groq
// (gpt-oss-120b) computes the nutrition estimate from the English text,
// which testing showed is dramatically more accurate than Hebrew input, for
// calories AND macros. Deploy with `wrangler deploy` after setting
// ALLOWED_ORIGIN, binding a KV namespace as RATE_LIMIT, and setting the
// GROQ_API_KEY and DEEPL_API_KEY secrets (see README.md for exact steps).

const GROQ_MODEL = 'openai/gpt-oss-120b';
const GROQ_TIMEOUT_MS = 10000;
const DEEPL_TIMEOUT_MS = 8000;
const RATE_LIMIT_PER_HOUR = 20;
const MAX_NAME_LEN = 80;

// weight_g/calories_per_100g force the model to compute them first, as a
// structured chain-of-thought, before the final numbers — this measurably
// improved accuracy and eliminated near-identical answers across different
// foods during testing. Macros are back in the schema: unlike Gemini
// flash-lite, Groq reliably produced correct per-food macro ratios once fed
// English input.
const RESPONSE_SCHEMA = {
  type: 'object',
  properties: {
    weight_g:          { type: 'number' },
    calories_per_100g: { type: 'number' },
    calories:          { type: 'number' },
    protein_g:         { type: 'number' },
    carbs_g:           { type: 'number' },
    fat_g:             { type: 'number' },
  },
  required: ['weight_g', 'calories_per_100g', 'calories', 'protein_g', 'carbs_g', 'fat_g'],
  additionalProperties: false,
};

const NUTRITION_SYSTEM_PROMPT = `You are a nutrition estimation assistant. Given a food description, estimate its nutrition. Work it out step by step in this order: 1) weight_g: typical serving weight in grams (use the weight stated in the description if given, otherwise a realistic typical portion for this specific food). 2) calories_per_100g: realistic calories per 100 grams for this specific food, must vary meaningfully between different foods, not a generic average. 3) calories: weight_g / 100 * calories_per_100g. 4) protein_g, carbs_g, fat_g: grams for the total weight_g, reflecting this specific foods real macronutrient profile - lean meats and eggs are protein-dominant with close to 0g carbs, grains and fruit are carb-dominant, oils/fats are almost entirely fat. Do not use similar ratios across different food types. Be precise and specific to this exact food, not a generic estimate.`;

function corsHeaders(env) {
  return {
    'Access-Control-Allow-Origin': env.ALLOWED_ORIGIN,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  };
}

function jsonResponse(body, status, env) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...corsHeaders(env) },
  });
}

function clamp(n, min, max) {
  n = Number(n);
  if (!Number.isFinite(n)) return null;
  return Math.min(max, Math.max(min, n));
}

// Fixed-window counter keyed by IP + current hour bucket; the key expires on
// its own via KV's TTL, so there's nothing to clean up. Any KV failure fails
// open (request is allowed) — this is an abuse deterrent, not an auth layer.
// 20/hour is unrelated to either upstream provider's own quota (Groq: 1000
// req/day free tier; DeepL: ~1M chars one-time credit, years of headroom at
// this app's usage) — it's sized for legitimate solo use regardless of
// which providers sit behind it.
async function checkRateLimit(env, ip) {
  if (!env.RATE_LIMIT) return true;
  const bucket = Math.floor(Date.now() / 3600000);
  const key = `rl:${ip}:${bucket}`;
  try {
    const current = parseInt((await env.RATE_LIMIT.get(key)) || '0', 10);
    if (current >= RATE_LIMIT_PER_HOUR) return false;
    await env.RATE_LIMIT.put(key, String(current + 1), { expirationTtl: 3600 });
    return true;
  } catch (e) {
    return true;
  }
}

// Step 1: dedicated (non-LLM) machine translation, not the estimating model
// itself — Gemini and Groq's chat models both proved unreliable at reading
// short Hebrew input directly. Free-tier DeepL keys (ending in ":fx") must
// use the api-free subdomain, not api.deepl.com.
async function translateToEnglish(env, name) {
  const url = 'https://api-free.deepl.com/v2/translate';
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), DEEPL_TIMEOUT_MS);

  let res;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `DeepL-Auth-Key ${env.DEEPL_API_KEY}`,
      },
      body: JSON.stringify({ text: [name], source_lang: 'HE', target_lang: 'EN-US' }),
      signal: controller.signal,
    });
  } catch (e) {
    throw { code: 'translation_failed' };
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    const errText = await res.text();
    console.error('DeepL API error', res.status, errText);
    throw { code: 'translation_failed' };
  }

  const data = await res.json();
  const english = data?.translations?.[0]?.text;
  if (!english || typeof english !== 'string') throw { code: 'translation_failed' };

  return english;
}

// Step 2: the actual nutrition estimate, from English text only.
async function estimateNutritionOnce(env, englishName) {
  const url = 'https://api.groq.com/openai/v1/chat/completions';
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), GROQ_TIMEOUT_MS);

  let res;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${env.GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: GROQ_MODEL,
        response_format: {
          type: 'json_schema',
          json_schema: { name: 'nutrition', strict: true, schema: RESPONSE_SCHEMA },
        },
        messages: [
          { role: 'system', content: NUTRITION_SYSTEM_PROMPT },
          { role: 'user', content: englishName },
        ],
      }),
      signal: controller.signal,
    });
  } catch (e) {
    if (e.name === 'AbortError') throw { code: 'timeout' };
    throw { code: 'ai_error' };
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    const errText = await res.text();
    console.error('Groq API error', res.status, errText);
    throw { code: res.status === 429 ? 'quota_exceeded' : 'ai_error' };
  }

  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content;
  if (!text) throw { code: 'ai_error' };

  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    throw { code: 'ai_error' };
  }

  const calories  = clamp(parsed.calories, 0, 9999);
  const protein_g = clamp(parsed.protein_g, 0, 999);
  const carbs_g   = clamp(parsed.carbs_g, 0, 999);
  const fat_g     = clamp(parsed.fat_g, 0, 999);
  if (calories === null || protein_g === null || carbs_g === null || fat_g === null) {
    throw { code: 'ai_error' };
  }

  return { calories, protein_g, carbs_g, fat_g };
}

function isAllZero(r) {
  return r.calories === 0 && r.protein_g === 0 && r.carbs_g === 0 && r.fat_g === 0;
}

// A "successful" (HTTP 200) all-zero response is a masked failure, not a
// real answer — no real food is exactly 0 kcal/0g everything. Retry once
// with the SAME already-translated English text (no need to re-translate);
// accept whatever the second attempt returns (even if also zero, e.g. a
// legitimately ~0-kcal item like plain water) rather than looping forever.
async function estimateNutrition(env, englishName) {
  const first = await estimateNutritionOnce(env, englishName);
  if (!isAllZero(first)) return first;
  console.error('Groq returned all-zero values, retrying once');
  return estimateNutritionOnce(env, englishName);
}

export default {
  async fetch(request, env) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders(env) });
    }
    if (request.method !== 'POST') {
      return jsonResponse({ error: 'method_not_allowed' }, 405, env);
    }

    let body;
    try {
      body = await request.json();
    } catch (e) {
      return jsonResponse({ error: 'invalid_body' }, 400, env);
    }

    const name = typeof body.name === 'string' ? body.name.trim().slice(0, MAX_NAME_LEN) : '';
    if (!name) {
      return jsonResponse({ error: 'missing_name' }, 400, env);
    }

    const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
    const allowed = await checkRateLimit(env, ip);
    if (!allowed) {
      return jsonResponse({ error: 'rate_limited' }, 429, env);
    }

    try {
      const english = await translateToEnglish(env, name);
      const result = await estimateNutrition(env, english);
      return jsonResponse(result, 200, env);
    } catch (e) {
      console.error(e);
      const code = e && e.code ? e.code : 'ai_error';
      const status = code === 'timeout' ? 504 : code === 'quota_exceeded' ? 429 : 502;
      return jsonResponse({ error: code }, status, env);
    }
  },
};
