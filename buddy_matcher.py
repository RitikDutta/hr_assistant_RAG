import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None


load_dotenv()


INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "employee-buddy-poc")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "employees")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
SEMANTIC_SCORE_THRESHOLD = float(os.getenv("SEMANTIC_SCORE_THRESHOLD", "0.72"))
TRUE_VALUES = {"1", "true", "yes", "y", "on"}
PLACEHOLDER_API_KEYS = {"x", "xx", "xxx", "...", "replace-me", "your-key-here"}
RERANK_WEIGHTS = {
    "semantic_score": 0.45,
    "skill_overlap": 0.20,
    "domain_match": 0.15,
    "department_match": 0.10,
    "interest_overlap": 0.05,
    "location_match": 0.05,
}


def is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def is_demo_mode_enabled() -> bool:
    return is_truthy(os.getenv("DEMO_MODE"))


def validate_api_key(value: str | None, label: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise RuntimeError(f"Set {label} in your environment.")
    if cleaned.lower() in PLACEHOLDER_API_KEYS or len(cleaned) < 10:
        raise RuntimeError(
            f"{label} looks like a placeholder. Use a real API key and pass it as an environment variable."
        )
    return cleaned


def employee_text(profile: dict[str, Any]) -> str:
    """Turn structured employee metadata into text for embedding and RAG context."""
    summary = str(profile.get("profile_text") or "").strip()
    lines = []
    if summary:
        lines.append(f"Summary: {summary}")

    lines.extend(
        [
            f"Name: {profile.get('name')}",
            f"Role: {profile.get('role')}",
            f"Domain: {profile.get('domain')}",
            f"Department: {profile.get('department')}",
            f"Location: {profile.get('location')}",
            f"Skills: {', '.join(profile.get('skills', []))}",
            f"Experience: {profile.get('experience_years', 0)} years",
            f"Interests: {', '.join(profile.get('interests', []))}",
            f"Previous projects: {', '.join(profile.get('previous_projects', []))}",
        ]
    )
    return "\n".join(lines)


def list_overlap(left: list[str], right: list[str]) -> float:
    left_set = {item.lower() for item in left}
    right_set = {item.lower() for item in right}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def matching_items(left: list[str], right: list[str]) -> list[str]:
    right_lookup = {str(item).lower(): str(item) for item in right}
    return sorted(right_lookup[item] for item in {str(item).lower() for item in left} & set(right_lookup))


def rerank_details(new_employee: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate["metadata"]

    semantic_score = float(candidate.get("semantic_score", 0.0))
    skill_score = list_overlap(new_employee.get("skills", []), metadata.get("skills", []))
    interest_score = list_overlap(new_employee.get("interests", []), metadata.get("interests", []))
    domain_score = 1.0 if new_employee.get("domain") == metadata.get("domain") else 0.0
    department_score = 1.0 if new_employee.get("department") == metadata.get("department") else 0.0
    location_score = 1.0 if new_employee.get("location") == metadata.get("location") else 0.0

    weighted_scores = {
        "semantic_score": round(semantic_score * RERANK_WEIGHTS["semantic_score"], 4),
        "skill_overlap": round(skill_score * RERANK_WEIGHTS["skill_overlap"], 4),
        "domain_match": round(domain_score * RERANK_WEIGHTS["domain_match"], 4),
        "department_match": round(department_score * RERANK_WEIGHTS["department_match"], 4),
        "interest_overlap": round(interest_score * RERANK_WEIGHTS["interest_overlap"], 4),
        "location_match": round(location_score * RERANK_WEIGHTS["location_match"], 4),
    }
    final_score = round(sum(weighted_scores.values()), 4)

    return {
        "raw_scores": {
            "semantic_score": round(semantic_score, 4),
            "skill_overlap": round(skill_score, 4),
            "domain_match": domain_score,
            "department_match": department_score,
            "interest_overlap": round(interest_score, 4),
            "location_match": location_score,
        },
        "matches": {
            "skills": matching_items(new_employee.get("skills", []), metadata.get("skills", [])),
            "interests": matching_items(new_employee.get("interests", []), metadata.get("interests", [])),
            "domain": new_employee.get("domain") == metadata.get("domain"),
            "department": new_employee.get("department") == metadata.get("department"),
            "location": new_employee.get("location") == metadata.get("location"),
        },
        "weights": RERANK_WEIGHTS,
        "weighted_scores": weighted_scores,
        "final_score": final_score,
    }


def rerank_score(new_employee: dict[str, Any], candidate: dict[str, Any]) -> float:
    return float(rerank_details(new_employee, candidate)["final_score"])


def pinecone_response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "to_json"):
        return json.loads(response.to_json())
    try:
        return dict(response)
    except (TypeError, ValueError):
        return {}


def response_value(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


def namespace_vector_count(stats: dict[str, Any], namespace: str) -> int:
    namespaces = stats.get("namespaces") or {}
    namespace_stats = namespaces.get(namespace) or {}
    return int(namespace_stats.get("vector_count") or 0)


def format_list(values: Any) -> str:
    if isinstance(values, list):
        items = [str(item).strip() for item in values if str(item).strip()]
    elif values:
        items = [str(values).strip()]
    else:
        items = []
    return ", ".join(items) if items else "Not provided"


def embedding_preview(values: list[float], limit: int = 5) -> str:
    preview = ", ".join(f"{float(value):.3f}" for value in values[:limit])
    suffix = ", ..." if len(values) > limit else ""
    return f"[{preview}{suffix}]"


def candidate_name(candidate: dict[str, Any]) -> str:
    return str(candidate.get("metadata", {}).get("name") or "Unknown employee")


def candidate_domain(candidate: dict[str, Any]) -> str:
    return str(candidate.get("metadata", {}).get("domain") or "Unknown domain")


def semantic_candidate_lines(candidates: list[dict[str, Any]], limit: int = 3) -> list[str]:
    if not candidates:
        return ["No similar employees found from semantic search."]
    return [
        f"{index}. {candidate_name(candidate)} - {candidate_domain(candidate)} - "
        f"Similarity Score: {float(candidate.get('semantic_score', 0.0)):.2f}"
        for index, candidate in enumerate(candidates[:limit], start=1)
    ]


def ranked_candidate_lines(candidates: list[dict[str, Any]], limit: int = 3) -> list[str]:
    if not candidates:
        return ["No candidates available for reranking."]
    return [
        f"{index}. {candidate_name(candidate)} - Final Score: {float(candidate.get('rerank_score', 0.0)):.2f}"
        for index, candidate in enumerate(candidates[:limit], start=1)
    ]


def semantic_filter() -> dict[str, Any]:
    return {"availability_for_buddy": {"$eq": True}}


def fallback_filter(new_employee: dict[str, Any]) -> dict[str, Any]:
    metadata_filter = semantic_filter()
    for key in ("department", "domain"):
        value = new_employee.get(key)
        if value:
            metadata_filter[key] = {"$eq": value}
    return metadata_filter


def embedding_values(values: list[float]) -> list[float]:
    return [float(value) for value in values]


def candidate_details(candidate: dict[str, Any], include_rerank: bool = False) -> dict[str, Any]:
    metadata = candidate.get("metadata", {})
    details = {
        "employee_id": candidate.get("employee_id") or metadata.get("employee_id"),
        "name": metadata.get("name"),
        "role": metadata.get("role"),
        "domain": metadata.get("domain"),
        "department": metadata.get("department"),
        "location": metadata.get("location"),
        "skills": metadata.get("skills", []),
        "interests": metadata.get("interests", []),
        "availability_for_buddy": metadata.get("availability_for_buddy"),
        "source": candidate.get("source"),
        "semantic_score": float(candidate.get("semantic_score", 0.0)),
        "metadata": metadata,
    }
    if include_rerank:
        details["rerank_score"] = float(candidate.get("rerank_score", 0.0))
        details["rerank_details"] = candidate.get("rerank_details", {})
    return details


def demo_step(title: str, lines: list[str]) -> dict[str, Any]:
    return {"title": title, "lines": lines}


def print_demo_step(step: dict[str, Any]) -> None:
    print("\n----------------------------------------")
    print(step["title"])
    print("----------------------------------------")
    for line in step["lines"]:
        print(line)


def print_demo_steps(steps: list[dict[str, Any]]) -> None:
    for step in steps:
        print_demo_step(step)


def print_assigned_buddy(buddy: dict[str, Any]) -> None:
    metadata = buddy["metadata"]

    print("\nAssigned buddy")
    print("--------------")
    print(f"Name: {metadata['name']} ({metadata['employee_id']})")
    print(f"Role: {metadata['role']}")
    print(f"Domain: {metadata['domain']}")
    print(f"Department: {metadata['department']}")
    print(f"Location: {metadata['location']}")
    print(f"Match source: {buddy['source']}")
    print(f"Semantic score: {buddy['semantic_score']:.3f}")
    print(f"Rerank score: {buddy['rerank_score']:.3f}")
    print(f"Reason: {buddy['llm_reason']}")


class BuddyMatcher:
    def __init__(self) -> None:
        google_api_key = validate_api_key(
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            "GEMINI_API_KEY or GOOGLE_API_KEY",
        )
        pinecone_api_key = validate_api_key(os.getenv("PINECONE_API_KEY"), "PINECONE_API_KEY")

        try:
            from google import genai
            from google.genai import types
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc

        self.types = types
        self.serverless_spec = ServerlessSpec
        self.gemini = genai.Client(api_key=google_api_key)
        self.pinecone = Pinecone(api_key=pinecone_api_key)
        self.embedding_dimensions = self.resolve_embedding_dimensions()

    def resolve_embedding_dimensions(self) -> int:
        if not self.pinecone.has_index(name=INDEX_NAME):
            return EMBEDDING_DIMENSIONS

        description = self.pinecone.describe_index(name=INDEX_NAME)
        dimension = getattr(description, "dimension", None)
        if dimension is None and hasattr(description, "get"):
            dimension = description.get("dimension")
        return int(dimension or EMBEDDING_DIMENSIONS)

    def ensure_index(self) -> None:
        if not self.pinecone.has_index(name=INDEX_NAME):
            self.pinecone.create_index(
                name=INDEX_NAME,
                dimension=self.embedding_dimensions,
                metric="cosine",
                spec=self.serverless_spec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1"),
                ),
                deletion_protection="disabled",
            )

        while not self.pinecone.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(2)

    def index(self):
        return self.pinecone.Index(INDEX_NAME)

    def embed(self, text: str) -> list[float]:
        response = self.gemini.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=self.types.EmbedContentConfig(output_dimensionality=self.embedding_dimensions),
        )
        return response.embeddings[0].values

    def seed_employees(self, employees: list[dict[str, Any]]) -> dict[str, Any]:
        self.ensure_index()
        vectors = []

        for employee in employees:
            metadata = {
                **employee,
                "profile_text": employee_text(employee),
            }
            vectors.append(
                {
                    "id": employee["employee_id"],
                    "values": self.embed(metadata["profile_text"]),
                    "metadata": metadata,
                }
            )

        upsert_response = self.index().upsert(vectors=vectors, namespace=NAMESPACE)
        expected_count = len({vector["id"] for vector in vectors})
        stats = self.wait_for_namespace_count(expected_count)
        namespace_count = namespace_vector_count(stats, NAMESPACE)
        total_count = int(stats.get("total_vector_count") or 0)
        upserted_count = response_value(upsert_response, "upserted_count")

        print(
            f"Seeded {len(vectors)} employee profiles into Pinecone index "
            f"'{INDEX_NAME}' namespace '{NAMESPACE}'. Pinecone namespace count: {namespace_count}."
        )
        return {
            "index_name": INDEX_NAME,
            "namespace": NAMESPACE,
            "upserted_count": int(upserted_count or len(vectors)),
            "namespace_vector_count": namespace_count,
            "total_vector_count": total_count,
        }

    def index_stats(self) -> dict[str, Any]:
        return pinecone_response_to_dict(self.index().describe_index_stats())

    def wait_for_namespace_count(self, minimum_count: int) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for attempt in range(6):
            stats = self.index_stats()
            if namespace_vector_count(stats, NAMESPACE) >= minimum_count:
                return stats
            if attempt < 5:
                time.sleep(1)
        return stats

    def semantic_candidates(
        self,
        new_employee: dict[str, Any],
        top_k: int = 5,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        query_response = self.index().query(
            vector=query_vector if query_vector is not None else self.embed(employee_text(new_employee)),
            top_k=top_k,
            namespace=NAMESPACE,
            include_metadata=True,
            filter=semantic_filter(),
        )

        candidates = []
        for match in query_response.matches:
            candidates.append(
                {
                    "employee_id": match.id,
                    "semantic_score": match.score,
                    "metadata": match.metadata,
                    "source": "semantic_search",
                }
            )
        return candidates

    def fallback_candidates(
        self,
        new_employee: dict[str, Any],
        top_k: int = 5,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        query_response = self.index().query(
            vector=query_vector if query_vector is not None else self.embed(employee_text(new_employee)),
            top_k=top_k,
            namespace=NAMESPACE,
            include_metadata=True,
            filter=fallback_filter(new_employee),
        )

        candidates = []
        for match in query_response.matches:
            candidates.append(
                {
                    "employee_id": match.id,
                    "semantic_score": match.score,
                    "metadata": match.metadata,
                    "source": "metadata_fallback",
                }
            )
        return candidates

    def assign_buddy(
        self,
        new_employee: dict[str, Any],
        demo_mode: bool = False,
        demo_step_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        profile_text = employee_text(new_employee)

        def add_demo_step(title: str, lines: list[str]) -> None:
            if not demo_mode:
                return
            step = demo_step(title, lines)
            steps.append(step)
            if demo_step_handler:
                demo_step_handler(step)

        if demo_mode:
            add_demo_step(
                "STEP 1: New employee profile received",
                [
                    f"Name: {new_employee.get('name') or 'Not provided'}",
                    f"Role: {new_employee.get('role') or 'Not provided'}",
                    f"Domain: {new_employee.get('domain') or 'Not provided'}",
                    f"Skills: {format_list(new_employee.get('skills', []))}",
                ],
            )
            add_demo_step(
                "STEP 2: Preparing profile for semantic search",
                [
                    "The employee profile is converted into text so it can be embedded and searched.",
                ],
            )

        query_embedding = self.embed(profile_text)

        add_demo_step(
            "STEP 3: Embedding generated",
            [
                "Embedding created successfully using Google Embedding model.",
                f"Embedding preview: {embedding_preview(query_embedding)}",
            ],
        )

        semantic_matches = self.semantic_candidates(new_employee, query_vector=query_embedding)
        candidates = semantic_matches

        add_demo_step(
            "STEP 4: Searching similar employees in Pinecone",
            [
                "Found top similar employees from the knowledgebase.",
                *semantic_candidate_lines(semantic_matches),
            ],
        )

        if not candidates or candidates[0]["semantic_score"] < SEMANTIC_SCORE_THRESHOLD:
            candidates = self.fallback_candidates(new_employee, query_vector=query_embedding)
            add_demo_step(
                f"STEP {len(steps) + 1}: Fallback mode activated",
                [
                    "Semantic search confidence was low, so the app used metadata filtering based on domain, department, and availability.",
                ],
            )

        if not candidates:
            raise RuntimeError("No available buddy found from semantic search or metadata fallback.")

        for candidate in candidates:
            candidate["rerank_score"] = rerank_score(new_employee, candidate)

        ranked_candidates = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)

        add_demo_step(
            f"STEP {len(steps) + 1}: Reranking candidates",
            [
                "Candidates are reranked using semantic similarity, domain match, skill overlap, department match, and location match.",
                *ranked_candidate_lines(ranked_candidates),
            ],
        )

        assigned_buddy = ranked_candidates[0]
        assigned_buddy["llm_reason"] = self.explain_match(new_employee, assigned_buddy, ranked_candidates[:3])
        add_demo_step(
            "FINAL RESULT:",
            [
                f"Assigned Buddy: {candidate_name(assigned_buddy)}",
                f"Reason: {assigned_buddy['llm_reason']}",
            ],
        )
        if demo_mode:
            assigned_buddy["demo_steps"] = steps
        return assigned_buddy

    def assign_buddy_implementation_demo(self, new_employee: dict[str, Any]) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        profile_text = employee_text(new_employee)

        def add_step(title: str, data: dict[str, Any]) -> None:
            steps.append({"title": title, "data": data})

        add_step(
            "STEP 1: New employee profile received",
            {
                "new_employee": new_employee,
            },
        )
        add_step(
            "STEP 2: Profile converted for semantic search",
            {
                "profile_text": profile_text,
            },
        )

        query_embedding = self.embed(profile_text)
        add_step(
            "STEP 3: Embedding generated",
            {
                "status": "done",
                "model": EMBEDDING_MODEL,
                "dimensions": len(query_embedding),
                "vector": embedding_values(query_embedding),
            },
        )

        semantic_matches = self.semantic_candidates(new_employee, query_vector=query_embedding)
        candidates = semantic_matches
        add_step(
            "STEP 4: Pinecone semantic search completed",
            {
                "index": INDEX_NAME,
                "namespace": NAMESPACE,
                "top_k": 5,
                "filter": semantic_filter(),
                "semantic_score_threshold": SEMANTIC_SCORE_THRESHOLD,
                "candidates": [candidate_details(candidate) for candidate in semantic_matches],
            },
        )

        fallback_used = False
        if not candidates or candidates[0]["semantic_score"] < SEMANTIC_SCORE_THRESHOLD:
            fallback_used = True
            candidates = self.fallback_candidates(new_employee, query_vector=query_embedding)
            add_step(
                "STEP 5: Metadata fallback search completed",
                {
                    "reason": "No semantic candidate was found or the top semantic score was below the configured threshold.",
                    "filter": fallback_filter(new_employee),
                    "candidates": [candidate_details(candidate) for candidate in candidates],
                },
            )

        if not candidates:
            raise RuntimeError("No available buddy found from semantic search or metadata fallback.")

        for candidate in candidates:
            details = rerank_details(new_employee, candidate)
            candidate["rerank_details"] = details
            candidate["rerank_score"] = details["final_score"]

        ranked_candidates = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)
        add_step(
            f"STEP {len(steps) + 1}: Candidate reranking completed",
            {
                "formula": (
                    "semantic_score * 0.45 + skill_overlap * 0.20 + domain_match * 0.15 + "
                    "department_match * 0.10 + interest_overlap * 0.05 + location_match * 0.05"
                ),
                "weights": RERANK_WEIGHTS,
                "ranked_candidates": [
                    candidate_details(candidate, include_rerank=True) for candidate in ranked_candidates
                ],
            },
        )

        assigned_buddy = ranked_candidates[0]
        assigned_buddy["llm_reason"] = self.explain_match(new_employee, assigned_buddy, ranked_candidates[:3])
        add_step(
            "FINAL RESULT",
            {
                "fallback_used": fallback_used,
                "assigned_buddy": candidate_details(assigned_buddy, include_rerank=True),
                "reason_model": GEMINI_MODEL,
                "reason": assigned_buddy["llm_reason"],
            },
        )

        assigned_buddy["implementation_steps"] = steps
        return assigned_buddy

    def explain_match(
        self,
        new_employee: dict[str, Any],
        assigned_buddy: dict[str, Any],
        top_candidates: list[dict[str, Any]],
    ) -> str:
        context = "\n\n".join(
            [
                f"Candidate: {candidate['metadata']['name']}\n"
                f"Role: {candidate['metadata']['role']}\n"
                f"Domain: {candidate['metadata']['domain']}\n"
                f"Skills: {', '.join(candidate['metadata'].get('skills', []))}\n"
                f"Location: {candidate['metadata'].get('location')}\n"
                f"Semantic score: {candidate['semantic_score']:.3f}\n"
                f"Rerank score: {candidate['rerank_score']:.3f}"
                for candidate in top_candidates
            ]
        )

        prompt = f"""
You are helping HR assign an onboarding buddy.

New employee:
{employee_text(new_employee)}

Top matched existing employees:
{context}

Assigned buddy: {assigned_buddy['metadata']['name']}

In 2 short sentences, explain why this buddy is the best practical match.
"""
        response = self.gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return response.text.strip()


def read_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def seed_command(args: argparse.Namespace) -> None:
    matcher = BuddyMatcher()
    matcher.seed_employees(read_json(args.employees))


def run_assign(new_employee_path: str, demo_mode: bool = False) -> None:
    matcher = BuddyMatcher()
    effective_demo_mode = demo_mode or is_demo_mode_enabled()
    buddy = matcher.assign_buddy(
        read_json(new_employee_path),
        demo_mode=effective_demo_mode,
        demo_step_handler=print_demo_step if effective_demo_mode else None,
    )

    if effective_demo_mode:
        return

    print_assigned_buddy(buddy)


def assign_command(args: argparse.Namespace) -> None:
    run_assign(args.new_employee, demo_mode=args.demo)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Buddy assignment POC using Gemini embeddings, Pinecone, and RAG.")
    subparsers = parser.add_subparsers(required=True)

    seed_parser = subparsers.add_parser("seed", help="Embed and upsert existing employee profiles.")
    seed_parser.add_argument("--employees", default="employees.json")
    seed_parser.set_defaults(func=seed_command)

    assign_parser = subparsers.add_parser("assign", help="Assign a buddy for a new employee profile.")
    assign_parser.add_argument("--new-employee", default="new_employee.json")
    assign_parser.add_argument("--demo", action="store_true", help="Show a high-level demo-friendly process overview.")
    assign_parser.set_defaults(func=assign_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except RuntimeError as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
