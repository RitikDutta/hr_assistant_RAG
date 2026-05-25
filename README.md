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

Use real keys here. Placeholder values like `x`, `xx`, or `...` will be rejected.

## Run CLI

Seed existing employees into Pinecone:

```bash
python buddy_matcher.py seed --employees employees.json
```

Assign a buddy for the sample new employee:

```bash
python buddy_matcher.py assign --new-employee new_employee.json
```

## Run Demo Mode

Use demo mode for a meeting-friendly process overview:

```bash
python app.py --demo
```

You can also run the existing CLI command with `--demo`:

```bash
python buddy_matcher.py assign --new-employee new_employee.json --demo
```

Or enable it from `.env`:

```bash
DEMO_MODE=true
```

Demo mode shows only the high-level story: new employee received, profile prepared, embedding preview, Pinecone search, top 3 candidates, reranked top 3, and the final buddy.

For the web app, `POST /api/assign-demo` returns the normal buddy response plus a simple `steps` array.

For implementation-level debugging, use `POST /api/assign-implementation-demo` or `POST /api/assign-full-demo`. This returns the complete process trace, including the generated embedding vector, Pinecone query details, fallback details when used, and per-candidate reranking scores.

## Run with Docker

Build and run locally:

```bash
docker build -t employee-buddy-poc .
docker run --env-file .env -p 8080:8080 employee-buddy-poc
```

For Cloud Run, set `GEMINI_API_KEY` and `PINECONE_API_KEY` as service environment variables or secrets. The local `.env` file is not copied into the Docker image.

## Cloud Build Trigger

Use `cloudbuild.yaml` as the build configuration file in the trigger settings. By default it deploys Cloud Run service `employee-buddy-poc` in `us-central1`; change `_SERVICE_NAME` or `_REGION` in the trigger substitutions if your Cloud Run service uses a different name or region.

## Run Frontend

Start the basic Flask frontend:

```bash
python web_app.py
```

Open:

```text
http://127.0.0.1:8080
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
