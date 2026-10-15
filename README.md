# Dawaak AI Assistant API

FastAPI service that understands a user's pharmacy request (intent + entities)
via DeepSeek and returns structured JSON. It does NOT search databases or
invent prices/pharmacies — the backend (e.g. Supabase) does that.

## Endpoints

- `GET /` — liveness: `{"status":"success",...}`
- `GET /health` — health check for Render: `{"status":"ok","version":"1.1"}`
- `POST /ai/assistant` — body `{"message":"وين بلاقي بنادول؟"}`
  - Optional header `X-API-Key` — required only when `DAWAAK_API_KEY` is set.
  - Returns `{"source":"ai", intent, drug_name, symptoms, requires_location,
    requested_information, filters, result_limit, is_emergency,
    raw_query, reply, guidance}`.
  - Errors: `422` bad input, `401` bad API key, `500` config, `503` AI busy.

Example (`search_medicine`):

```json
{
  "source": "ai",
  "intent": "search_medicine",
  "drug_name": "بنادول",
  "requires_location": true,
  "requested_information": ["availability", "pharmacies", "distance"],
  "result_limit": 3,
  "is_emergency": false,
  "reply": "...",
  "guidance": null
}
```

## Env vars

| Var | Required | Default | Purpose |
|---|---|---|---|
| `DEEPSEEK_API_KEY` (or `AI_API_KEY`) | yes | — | DeepSeek key |
| `AI_BASE_URL` | no | `https://api.deepseek.com` | API base |
| `AI_MODEL` / `DEEPSEEK_MODEL` | no | `deepseek-chat` | Model |
| `AI_TIMEOUT` | no | `15` | Seconds per AI call |
| `AI_MAX_RETRIES` | no | `0` | OpenAI client retries (loop handles retry) |
| `DAWAAK_API_KEY` (or `DAWAY_API_KEY`, `API_KEY`) | no | unset = open | Gate `POST /ai/assistant` |

## Run locally

```powershell
py -m venv venv; venv\Scripts\Activate
venv\Scripts\pip install -r requirements.txt
copy .env.example .env  # then fill DEEPSEEK_API_KEY
py -m uvicorn app:app --port 8000
# docs: http://127.0.0.1:8000/docs
```

## Docker / Render

```powershell
docker build -t dawaak-ai .
docker run -p 10000:10000 -e DEEPSEEK_API_KEY=... dawaak-ai
```

Render: set env vars in dashboard, health check path `/health`.

## Backend wiring (Supabase sketch)

- `drug_name + result_limit` → `medicines` + `availability(pharmacy_id, price, in_stock)`
- `requires_location` → `pharmacies(lat, lng)` with PostGIS nearest + distance
- `find_alternative` → `alternatives`, `drug_info` → `medicines` details
- `reply/guidance` is chat text; medicine facts always come from the DB.
