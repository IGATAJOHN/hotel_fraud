# FAQ Bot (Groq or Gemini)

FAQ bot for hotel guest questions. Uses FAQ data from `app/core/data/faqs.json` and returns smart, human-like answers. **Uses Groq if `GROQ_API_KEY` is set, otherwise Gemini** (so you can use free Groq without hitting Gemini quotas).

## Requirements

- **Python 3.10, 3.11, or 3.12.** The project uses CatBoost (for fraud detection); CatBoost does not support Python 3.14 yet and will fail to build. Use a 3.10–3.12 interpreter for the venv.

## How to run

### 1. Install dependencies

From the **project root** (`hotel_fraud/`):

```bash
pip install -r requirements.txt
```

If you see **"Failed building wheel for catboost"**, switch to Python 3.11 or 3.12, then recreate the venv and install again:

```bash
# Example: create a new venv with Python 3.12 (if installed via pyenv, Homebrew, or python.org)
python3.12 -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Set an API key (Groq or Gemini)

**Option A – Groq (recommended, free tier):**

1. Get a key at [console.groq.com](https://console.groq.com).
2. `export GROQ_API_KEY="your-groq-key"`
3. Optional: `export GROQ_FAQ_MODEL=llama-3.1-8b-instant` (default is `llama-3.3-70b-versatile`).

**Option B – Gemini:**

1. Get a key at [Google AI Studio](https://aistudio.google.com/apikey).
2. `export GEMINI_API_KEY="your-gemini-key"`
3. Optional: `export GEMINI_FAQ_MODEL=gemini-1.5-flash` (default is `gemini-2.0-flash`).

If both are set, **Groq is used**. If you get **429** / quota errors, the API returns **503** with a `Retry-After` header; try again after that time or switch to the other provider.

### 3. Start the API

From the **project root**:

```bash
uvicorn main:app --reload
```

The FAQ bot is mounted on the same app. Once the server is running:

- **POST** `http://localhost:8000/faq/ask` — ask a question
- **GET** `http://localhost:8000/faq/hotels` — list hotel IDs with FAQs

### Example: ask a question

```bash
curl -X POST http://localhost:8000/faq/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What time is check-in?", "hotel_id": "hotel_001"}'
```

Omit `hotel_id` to use FAQs from all hotels:

```bash
curl -X POST http://localhost:8000/faq/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Do you have breakfast?"}'
```

### Docs

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

**Note:** If neither `GROQ_API_KEY` nor `GEMINI_API_KEY` is set, `POST /faq/ask` returns 503 with a message that the bot is not configured.
