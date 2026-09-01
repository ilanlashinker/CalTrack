// Cloudflare Worker: proxies food-name -> AI nutrition estimate requests to
// Gemini, so the API key never reaches the browser. Deploy with `wrangler
// deploy` after setting the ALLOWED_ORIGIN var, the GEMINI_API_KEY secret,
// and binding a KV namespace as RATE_LIMIT (see README.md for exact steps).

const GEMINI_MODEL = 'gemini-3.5-flash-lite';
const GEMINI_TIMEOUT_MS = 20000;
const RATE_LIMIT_PER_HOUR = 20;
const MAX_NAME_LEN = 80;

const RESPONSE_SCHEMA = {
  type: 'OBJECT',
  properties: {
    calories:  { type: 'NUMBER' },
    protein_g: { type: 'NUMBER' },
    carbs_g:   { type: 'NUMBER' },
    fat_g:     { type: 'NUMBER' },
  },
  required: ['calories', 'protein_g', 'carbs_g', 'fat_g'],
};

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

async function callGeminiOnce(env, name) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${env.GEMINI_API_KEY}`;
  const prompt = `Estimate realistic average nutrition values for ONE typical serving/unit of this food, as commonly consumed. Food name (may be in Hebrew or English): "${name}". Return calories (kcal), and grams of protein, carbs, and fat for that single unit.`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), GEMINI_TIMEOUT_MS);

  let res;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: {
          responseMimeType: 'application/json',
          responseSchema: RESPONSE_SCHEMA,
          // gemini-3.6-flash is a thinking model with a variable thinking
          // token count by default ('medium'), which is what caused the
          // wild latency swings (0.9s-32s) we measured. 'low' cuts that
          // down without going as far as 'minimal', where we saw degenerate
          // all-zero output more often. The retry-on-zero guard below is
          // the actual correctness safety net regardless of this setting.
          thinkingConfig: { thinkingLevel: 'low' },
        },
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
    console.error('Gemini API error', res.status, errText);
    throw { code: 'ai_error' };
  }

  const data = await res.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
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
// real answer — no real food is exactly 0 kcal/0g everything. Retry once;
// accept whatever the second attempt returns (even if also zero, e.g. a
// legitimately ~0-kcal item like plain water) rather than looping forever.
async function callGemini(env, name) {
  const first = await callGeminiOnce(env, name);
  if (!isAllZero(first)) return first;
  console.error('Gemini returned all-zero values, retrying once');
  return callGeminiOnce(env, name);
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
      const result = await callGemini(env, name);
      return jsonResponse(result, 200, env);
    } catch (e) {
      console.error(e);
      const code = e && e.code ? e.code : 'ai_error';
      const status = code === 'timeout' ? 504 : 502;
      return jsonResponse({ error: code }, status, env);
    }
  },
};
