import re
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from buddy_matcher import BuddyMatcher, read_json


app = Flask(__name__)

BASE_DIR = Path(__file__).parent
EMPLOYEES_FILE = BASE_DIR / "employees.json"
NEW_EMPLOYEE_FILE = BASE_DIR / "new_employee.json"

SKILL_ALIASES = [
    (("machine learning",), "Machine Learning"),
    (("ai/ml", "ai ml", "artificial intelligence"), "AI/ML"),
    (("vector databases", "vector database"), "Vector Databases"),
    (("fastapi",), "FastAPI"),
    (("postgresql", "postgres"), "PostgreSQL"),
    (("apis", "api"), "APIs"),
    (("mlops",), "MLOps"),
    (("rag",), "RAG"),
    (("embeddings", "embedding"), "Embeddings"),
    (("pinecone",), "Pinecone"),
    (("python",), "Python"),
    (("sql",), "SQL"),
    (("dashboards", "dashboard"), "Dashboards"),
    (("experimentation",), "Experimentation"),
    (("aws",), "AWS"),
    (("terraform",), "Terraform"),
    (("kubernetes",), "Kubernetes"),
    (("observability",), "Observability"),
]

DOMAIN_SIGNALS = {
    "AI/ML": ("ai/ml", "ai ml", "machine learning", "mlops", "rag", "embedding", "pinecone", "vector"),
    "HR Tech": ("hr", "onboarding", "employee self-service", "people systems", "process automation"),
    "Infrastructure": ("aws", "cloud", "kubernetes", "terraform", "observability", "platform"),
    "Analytics": ("analytics", "sql", "dashboard", "experimentation", "retention analysis"),
}

DEPARTMENT_BY_DOMAIN = {
    "AI/ML": "AI Team",
    "HR Tech": "People Systems",
    "Infrastructure": "Platform",
    "Analytics": "Product",
}

ROLE_SIGNALS = [
    (("backend", "fastapi", "api"), "Backend Engineer"),
    (("data scientist", "data science"), "Data Scientist"),
    (("machine learning engineer", "ml engineer", "mlops"), "Machine Learning Engineer"),
    (("cloud", "kubernetes", "terraform", "platform"), "Cloud Platform Engineer"),
    (("analyst", "analytics", "dashboard"), "Product Analyst"),
]


def api_error(message: str, status_code: int = 400):
    return jsonify({"detail": message}), status_code


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Buddy Assignment POC</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --ink: #1f2933;
      --muted: #607080;
      --line: #d8dee6;
      --primary: #166b5f;
      --primary-dark: #0f5148;
      --blue: #275f9d;
      --amber: #9a6400;
      --error: #a83232;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0;
    }

    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 22px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 16px;
    }

    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
      letter-spacing: 0;
    }

    .meta {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
      margin: 7px 0 0;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(320px, 460px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 8px 24px rgba(31, 41, 51, 0.06);
    }

    .panel-title {
      margin: 0 0 14px;
      font-size: 17px;
      line-height: 1.2;
      letter-spacing: 0;
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    label {
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }

    label.wide {
      grid-column: 1 / -1;
    }

    input,
    textarea {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      font: inherit;
      font-size: 14px;
      padding: 10px 11px;
      outline: none;
    }

    textarea {
      min-height: 68px;
      resize: vertical;
    }

    textarea.profile-input {
      min-height: 190px;
      line-height: 1.5;
    }

    input:focus,
    textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(22, 107, 95, 0.14);
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }

    button {
      border: 0;
      border-radius: 6px;
      cursor: pointer;
      font: inherit;
      font-size: 14px;
      font-weight: 650;
      min-height: 40px;
      padding: 10px 14px;
    }

    .primary {
      background: var(--primary);
      color: #ffffff;
    }

    .primary:hover {
      background: var(--primary-dark);
    }

    .secondary {
      background: #e9eef3;
      color: var(--ink);
    }

    .secondary:hover {
      background: #dde5ed;
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.65;
    }

    .status {
      margin-top: 14px;
      min-height: 22px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
    }

    .status.error {
      color: var(--error);
    }

    .result-empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      display: grid;
      min-height: 260px;
      place-items: center;
      padding: 24px;
      text-align: center;
    }

    .buddy-name {
      margin: 0;
      font-size: 30px;
      line-height: 1.1;
      letter-spacing: 0;
    }

    .buddy-role {
      color: var(--muted);
      margin: 8px 0 18px;
      font-size: 15px;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }

    .metric {
      border-left: 4px solid var(--blue);
      background: #f4f8fc;
      border-radius: 6px;
      padding: 11px 12px;
      min-width: 0;
    }

    .metric:nth-child(2) {
      border-left-color: var(--primary);
      background: #f1f8f6;
    }

    .metric:nth-child(3) {
      border-left-color: var(--amber);
      background: #fbf7ed;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }

    .metric strong {
      display: block;
      font-size: 18px;
      overflow-wrap: anywhere;
    }

    .details {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 18px;
    }

    .detail {
      border-top: 1px solid var(--line);
      padding-top: 10px;
      min-width: 0;
    }

    .detail span {
      color: var(--muted);
      display: block;
      font-size: 12px;
      margin-bottom: 4px;
    }

    .detail strong {
      display: block;
      font-size: 14px;
      overflow-wrap: anywhere;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin: 8px 0 18px;
    }

    .chip {
      background: #edf1f5;
      border: 1px solid #dce4ec;
      border-radius: 999px;
      color: #334455;
      font-size: 12px;
      line-height: 1;
      padding: 7px 9px;
    }

    .reason {
      border-top: 1px solid var(--line);
      color: #344554;
      font-size: 15px;
      line-height: 1.55;
      margin: 0;
      padding-top: 14px;
    }

    @media (max-width: 860px) {
      main {
        width: min(100vw - 24px, 680px);
        padding: 18px 0;
      }

      header,
      .layout {
        display: block;
      }

      header .actions {
        margin-top: 14px;
      }

      .panel {
        margin-bottom: 14px;
      }
    }

    @media (max-width: 560px) {
      .grid,
      .metrics,
      .details {
        grid-template-columns: 1fr;
      }

      h1 {
        font-size: 24px;
      }

      .buddy-name {
        font-size: 25px;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Buddy Assignment</h1>
        <p class="meta">Semantic search + reranking + Gemini explanation</p>
      </div>
      <div class="actions">
        <button class="secondary" id="seedButton" type="button">Seed Employee KB</button>
        <button class="secondary" id="sampleButton" type="button">Load Sample</button>
      </div>
    </header>

    <section class="layout">
      <form class="panel" id="employeeForm">
        <h2 class="panel-title">New Employee</h2>
        <label>
          Employee Summary
          <textarea
            class="profile-input"
            name="profile_text"
            placeholder="New employee with 2 years of experience in Python and AI ML, projects are RAG document parser"
            required
          ></textarea>
        </label>
        <div class="actions">
          <button class="primary" id="assignButton" type="submit">Assign Buddy</button>
        </div>
        <div class="status" id="status"></div>
      </form>

      <section class="panel" aria-live="polite">
        <h2 class="panel-title">Assignment Result</h2>
        <div id="result" class="result-empty">No buddy assigned yet.</div>
      </section>
    </section>
  </main>

  <script>
    const form = document.querySelector("#employeeForm");
    const statusEl = document.querySelector("#status");
    const resultEl = document.querySelector("#result");
    const assignButton = document.querySelector("#assignButton");
    const seedButton = document.querySelector("#seedButton");
    const sampleButton = document.querySelector("#sampleButton");

    function setStatus(message, isError = false) {
      statusEl.textContent = message;
      statusEl.classList.toggle("error", isError);
    }

    function employeeFromForm() {
      return {
        profile_text: String(form.elements.profile_text.value).trim(),
      };
    }

    function fillForm(employee) {
      form.elements.profile_text.value = employee.profile_text || "";
    }

    function formatScore(score) {
      return Number(score || 0).toFixed(3);
    }

    function renderResult(data) {
      const skills = (data.skills || [])
        .map((skill) => `<span class="chip">${escapeHtml(skill)}</span>`)
        .join("");

      resultEl.className = "";
      resultEl.innerHTML = `
        <h3 class="buddy-name">${escapeHtml(data.name)}</h3>
        <p class="buddy-role">${escapeHtml(data.role)} · ${escapeHtml(data.employee_id)}</p>
        <div class="metrics">
          <div class="metric"><span>Semantic</span><strong>${formatScore(data.semantic_score)}</strong></div>
          <div class="metric"><span>Rerank</span><strong>${formatScore(data.rerank_score)}</strong></div>
          <div class="metric"><span>Source</span><strong>${escapeHtml(data.match_source)}</strong></div>
        </div>
        <div class="details">
          <div class="detail"><span>Domain</span><strong>${escapeHtml(data.domain)}</strong></div>
          <div class="detail"><span>Department</span><strong>${escapeHtml(data.department)}</strong></div>
          <div class="detail"><span>Location</span><strong>${escapeHtml(data.location)}</strong></div>
          <div class="detail"><span>Experience</span><strong>${escapeHtml(String(data.experience_years))} years</strong></div>
        </div>
        <div class="chips">${skills}</div>
        <p class="reason">${escapeHtml(data.reason)}</p>
      `;
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function readJsonResponse(response) {
      const contentType = response.headers.get("content-type") || "";
      const rawBody = await response.text();

      if (contentType.includes("application/json")) {
        return rawBody ? JSON.parse(rawBody) : {};
      }

      const message = response.ok
        ? "Server returned a non-JSON response."
        : `Server returned ${response.status}. Check the Flask console for details.`;
      throw new Error(message);
    }

    async function postJson(url, payload = {}) {
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const data = await readJsonResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || "Request failed");
      }
      return data;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      assignButton.disabled = true;
      setStatus("Finding best buddy...");

      try {
        const data = await postJson("/api/assign", employeeFromForm());
        renderResult(data);
        setStatus("Buddy assigned.");
      } catch (error) {
        setStatus(error.message, true);
      } finally {
        assignButton.disabled = false;
      }
    });

    seedButton.addEventListener("click", async () => {
      seedButton.disabled = true;
      setStatus("Seeding employee knowledgebase...");

      try {
        const data = await postJson("/api/seed");
        const details = [
          `Seeded ${data.count} employees into Pinecone.`,
          `Index: ${data.index_name}.`,
          `Namespace: ${data.namespace}.`,
        ];
        if (Number.isFinite(Number(data.namespace_vector_count))) {
          details.push(`Pinecone reports ${data.namespace_vector_count} vectors in that namespace.`);
        }
        setStatus(details.join(" "));
      } catch (error) {
        setStatus(error.message, true);
      } finally {
        seedButton.disabled = false;
      }
    });

    sampleButton.addEventListener("click", async () => {
      try {
        const response = await fetch("/api/sample-new-employee");
        fillForm(await readJsonResponse(response));
        setStatus("Sample loaded.");
      } catch (error) {
        setStatus(error.message, true);
      }
    });

    sampleButton.click();
  </script>
</body>
</html>
"""


@app.get("/")
def home() -> str:
    return HTML


@app.get("/api/sample-new-employee")
def sample_new_employee():
    try:
        return jsonify({"profile_text": employee_profile_summary(read_json(str(NEW_EMPLOYEE_FILE)))})
    except Exception as exc:
        return api_error(str(exc), 500)


@app.post("/api/seed")
def seed_employees():
    try:
        employees = read_json(str(EMPLOYEES_FILE))
        seed_result = BuddyMatcher().seed_employees(employees)
    except RuntimeError as exc:
        return api_error(str(exc), 400)
    except Exception as exc:
        return api_error(str(exc), 500)
    return jsonify({"status": "ok", "count": len(employees), **seed_result})


@app.post("/api/assign")
def assign_buddy():
    new_employee = request.get_json(silent=True) or {}
    try:
        buddy = BuddyMatcher().assign_buddy(normalize_employee(new_employee))
    except RuntimeError as exc:
        return api_error(str(exc), 400)
    except Exception as exc:
        return api_error(str(exc), 500)

    metadata = buddy["metadata"]
    return jsonify(
        {
            "employee_id": metadata["employee_id"],
            "name": metadata["name"],
            "role": metadata["role"],
            "domain": metadata["domain"],
            "department": metadata["department"],
            "location": metadata["location"],
            "skills": metadata.get("skills", []),
            "experience_years": metadata.get("experience_years", 0),
            "match_source": buddy["source"],
            "semantic_score": float(buddy["semantic_score"]),
            "rerank_score": float(buddy["rerank_score"]),
            "reason": buddy["llm_reason"],
        }
    )


def employee_profile_summary(employee: dict[str, Any]) -> str:
    name = str(employee.get("name") or "New employee").strip()
    role = str(employee.get("role") or "").strip()
    experience = employee.get("experience_years")
    skills = ", ".join(normalize_list(employee.get("skills", [])))
    projects = ", ".join(normalize_list(employee.get("previous_projects", [])))

    opening = name
    if role:
        opening += f" is a {role}"
    if experience not in (None, ""):
        opening += f" with {experience} years of experience"
    if skills:
        opening += f" in {skills}"

    sentences = [opening]
    if projects:
        sentences.append(f"Projects are {projects}")

    details = []
    for label, key in (("domain", "domain"), ("department", "department"), ("location", "location")):
        value = str(employee.get(key) or "").strip()
        if value:
            details.append(f"{label} is {value}")
    if details:
        sentences.append(", ".join(details))

    return ". ".join(sentences) + "."


def normalize_employee(employee: dict[str, Any]) -> dict[str, Any]:
    profile_text = clean_text(employee.get("profile_text") or employee.get("summary") or "")
    cleaned = parse_employee_summary(profile_text) if profile_text else {}

    for key, value in employee.items():
        if key in {"profile_text", "summary"} or not has_value(value):
            continue
        cleaned[key] = value

    if profile_text:
        cleaned["profile_text"] = profile_text

    for key in ("skills", "interests", "previous_projects"):
        cleaned[key] = normalize_list(cleaned.get(key, []))

    cleaned["experience_years"] = normalize_int(cleaned.get("experience_years"))
    cleaned["domain"] = cleaned.get("domain") or infer_domain(profile_text, cleaned.get("skills", []))
    cleaned["department"] = cleaned.get("department") or DEPARTMENT_BY_DOMAIN.get(cleaned["domain"], "")
    cleaned["role"] = cleaned.get("role") or infer_role(profile_text, cleaned["domain"])
    cleaned["employee_id"] = cleaned.get("employee_id") or "NEW001"
    cleaned["name"] = cleaned.get("name") or "New employee"
    cleaned["location"] = cleaned.get("location") or ""
    return cleaned


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def parse_employee_summary(text: str) -> dict[str, Any]:
    profile: dict[str, Any] = {"profile_text": text}

    name = extract_name(text)
    if name:
        profile["name"] = name

    experience = extract_experience(text)
    if experience is not None:
        profile["experience_years"] = experience

    skills = extract_skills(text)
    if skills:
        profile["skills"] = skills

    projects = extract_projects(text)
    if projects:
        profile["previous_projects"] = projects

    interests = extract_labeled_items(text, "interests?")
    if interests:
        profile["interests"] = interests

    location = extract_labeled_value(text, r"location|located in|based in")
    if location:
        profile["location"] = title_text(location)

    department = extract_labeled_value(text, "department")
    if department:
        profile["department"] = title_text(department)

    domain = canonical_domain(extract_labeled_value(text, "domain")) or infer_domain(text, skills)
    if domain:
        profile["domain"] = domain

    role = infer_role(text, domain)
    if role:
        profile["role"] = role

    return profile


def extract_experience(text: str) -> int | None:
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(float(match.group(1)))


def extract_name(text: str) -> str:
    match = re.search(
        r"\b(?:name is|named|called)\s+([A-Za-z][A-Za-z]*(?:\s+[A-Za-z][A-Za-z]*){0,3})",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"^\s*([A-Za-z][A-Za-z]*(?:\s+[A-Za-z][A-Za-z]*){0,2})\s+is\s+a\b",
            text,
            flags=re.IGNORECASE,
        )
    if not match:
        return ""
    name = title_text(match.group(1))
    return "" if name.lower() == "new employee" else name


def extract_skills(text: str) -> list[str]:
    lower_text = text.lower()
    skills: list[str] = []

    for aliases, canonical in SKILL_ALIASES:
        if any(phrase_in_text(lower_text, alias) for alias in aliases):
            append_unique(skills, canonical)

    skill_patterns = [
        r"\b(?:experience|experienced|skilled|skills?)\s+(?:in|with|are|include|includes)\s+(.+?)(?=(?:[,;.]?\s*(?:his|her|their)?\s*(?:previous\s+)?projects?\b)|[.;]|$)",
        r"\b(?:knows|uses|working with)\s+(.+?)(?=(?:[,;.]?\s*(?:his|her|their)?\s*(?:previous\s+)?projects?\b)|[.;]|$)",
    ]
    for pattern in skill_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            for item in split_items(match.group(1)):
                skill = canonical_skill(item)
                if skill:
                    append_unique(skills, skill)

    return skills


def extract_projects(text: str) -> list[str]:
    projects: list[str] = []
    pattern = (
        r"\b(?:previous\s+projects?|projects?|worked on|project work)\s*"
        r"(?:are|include|includes|were|is|:)?\s+"
        r"(.+?)(?=(?:[,;.]\s*(?:skills?|interests?|department|location|domain|role|experience)\b)|[.;]|$)"
    )
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        for project in split_items(match.group(1)):
            cleaned = clean_text(project).strip(" .,:;-")
            if cleaned:
                append_unique(projects, title_text(cleaned))
    return projects


def extract_labeled_items(text: str, label_pattern: str) -> list[str]:
    value = extract_labeled_value(text, label_pattern)
    return [title_text(item) for item in split_items(value)] if value else []


def extract_labeled_value(text: str, label_pattern: str) -> str:
    pattern = rf"\b(?:{label_pattern})\s*(?:is|are|include|includes|:)?\s+(.+?)(?=[,.;]|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return clean_text(match.group(1)) if match else ""


def infer_domain(text: str, skills: list[str]) -> str:
    haystack = f"{text} {' '.join(skills)}".lower()
    scores = {
        domain: sum(1 for signal in signals if phrase_in_text(haystack, signal))
        for domain, signals in DOMAIN_SIGNALS.items()
    }
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    return best_domain if best_score else ""


def canonical_domain(value: str) -> str:
    lower_value = value.lower()
    if not lower_value:
        return ""
    for domain, signals in DOMAIN_SIGNALS.items():
        if lower_value == domain.lower() or any(phrase_in_text(lower_value, signal) for signal in signals):
            return domain
    return title_text(value)


def infer_role(text: str, domain: str) -> str:
    lower_text = text.lower()
    explicit_role = extract_labeled_value(text, "role")
    if explicit_role:
        return title_text(explicit_role)

    for signals, role in ROLE_SIGNALS:
        if any(phrase_in_text(lower_text, signal) for signal in signals):
            return role
    if domain == "AI/ML":
        return "AI/ML Engineer"
    return ""


def canonical_skill(value: str) -> str:
    cleaned = clean_text(value).strip(" .,:;-")
    lower_value = cleaned.lower()
    if not lower_value:
        return ""
    for aliases, canonical in SKILL_ALIASES:
        if lower_value in aliases or any(phrase_in_text(lower_value, alias) for alias in aliases):
            return canonical
    if len(cleaned.split()) <= 3:
        return title_text(cleaned)
    return ""


def normalize_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def split_items(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"\s*,\s*|\s*;\s*|\s+and\s+", value) if item.strip()]


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def title_text(value: str) -> str:
    acronyms = {"ai", "ml", "rag", "api", "apis", "aws", "sql", "hr"}
    words = []
    for word in clean_text(value).split():
        normalized = word.strip()
        if normalized.lower() in acronyms:
            words.append(normalized.upper())
        else:
            words.append(normalized[:1].upper() + normalized[1:].lower())
    return " ".join(words)


def phrase_in_text(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.lower()).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text.lower()))


def append_unique(items: list[str], value: str) -> None:
    if value and value.lower() not in {item.lower() for item in items}:
        items.append(value)


def has_value(value: Any) -> bool:
    return value is not None and value != "" and value != []


@app.errorhandler(HTTPException)
def handle_http_error(error: HTTPException):
    if request.path.startswith("/api/"):
        return api_error(error.description, error.code or 500)
    return error


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    if request.path.startswith("/api/"):
        return api_error(str(error), 500)
    raise error


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8800, debug=False)
