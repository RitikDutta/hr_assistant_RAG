# Buddy Assignment RAG POC

A small proof of concept for assigning an onboarding buddy to a new employee using:

- Google Gen AI embeddings
- Pinecone vector search
- A simple reranking formula
- Gemini as the reasoning layer for the final explanation
- Metadata fallback when semantic search is weak

## Flow

1. Existing employee profiles are embedded with `gemini-embedding-001`.
2. The embeddings and employee metadata are stored in Pinecone.
3. A new employee profile is embedded and searched against Pinecone.
4. Available employees are reranked by semantic score, skills, domain, department, interests, and location.
5. The highest ranked employee is assigned as the buddy.
6. If the top semantic match is below `SEMANTIC_SCORE_THRESHOLD`, the app falls back to Pinecone metadata filtering by department and domain.
7. Gemini generates a short HR-friendly explanation using the top matched candidates as context.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

```bash
GEMINI_API_KEY=...
PINECONE_API_KEY=...
```

## Run CLI

Seed existing employees into Pinecone:

```bash
python buddy_matcher.py seed --employees employees.json
```

Assign a buddy for the sample new employee:

```bash
python buddy_matcher.py assign --new-employee new_employee.json
```

## Run Frontend

Start the basic Flask frontend:

```bash
python web_app.py
```

Open:

```text
http://127.0.0.1:8000
```

Use **Seed Employee KB** once to load `employees.json` into Pinecone, then write one short HR summary for the new employee and use **Assign Buddy**.

## Customize

- Add or edit profiles in `employees.json`.
- Add a new hire profile in `new_employee.json`.
- Tune `SEMANTIC_SCORE_THRESHOLD` in `.env`.
- Use `GOOGLE_EMBEDDING_MODEL` to switch embedding models, as long as `EMBEDDING_DIMENSIONS` matches the Pinecone index dimension.

## Reranking Formula

```text
final_score =
  semantic_score * 0.45 +
  skill_overlap * 0.20 +
  domain_match * 0.15 +
  department_match * 0.10 +
  interest_overlap * 0.05 +
  location_match * 0.05
```

This is intentionally simple for a POC. It makes the semantic match important while still favoring practical buddy signals like shared skills, team context, interests, and location.
