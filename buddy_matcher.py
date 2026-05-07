import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

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


def rerank_score(new_employee: dict[str, Any], candidate: dict[str, Any]) -> float:
    semantic_score = float(candidate.get("semantic_score", 0.0))
    metadata = candidate["metadata"]

    skill_score = list_overlap(new_employee.get("skills", []), metadata.get("skills", []))
    interest_score = list_overlap(new_employee.get("interests", []), metadata.get("interests", []))
    domain_score = 1.0 if new_employee.get("domain") == metadata.get("domain") else 0.0
    department_score = 1.0 if new_employee.get("department") == metadata.get("department") else 0.0
    location_score = 1.0 if new_employee.get("location") == metadata.get("location") else 0.0

    return round(
        (semantic_score * 0.45)
        + (skill_score * 0.20)
        + (domain_score * 0.15)
        + (department_score * 0.10)
        + (interest_score * 0.05)
        + (location_score * 0.05),
        4,
    )


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


class BuddyMatcher:
    def __init__(self) -> None:
        google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")

        if not google_api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment.")
        if not pinecone_api_key:
            raise RuntimeError("Set PINECONE_API_KEY in your environment.")

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

    def semantic_candidates(self, new_employee: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
        query_response = self.index().query(
            vector=self.embed(employee_text(new_employee)),
            top_k=top_k,
            namespace=NAMESPACE,
            include_metadata=True,
            filter={"availability_for_buddy": {"$eq": True}},
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

    def fallback_candidates(self, new_employee: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
        metadata_filter: dict[str, Any] = {"availability_for_buddy": {"$eq": True}}
        for key in ("department", "domain"):
            value = new_employee.get(key)
            if value:
                metadata_filter[key] = {"$eq": value}

        query_response = self.index().query(
            vector=self.embed(employee_text(new_employee)),
            top_k=top_k,
            namespace=NAMESPACE,
            include_metadata=True,
            filter=metadata_filter,
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

    def assign_buddy(self, new_employee: dict[str, Any]) -> dict[str, Any]:
        candidates = self.semantic_candidates(new_employee)

        if not candidates or candidates[0]["semantic_score"] < SEMANTIC_SCORE_THRESHOLD:
            candidates = self.fallback_candidates(new_employee)

        if not candidates:
            raise RuntimeError("No available buddy found from semantic search or metadata fallback.")

        for candidate in candidates:
            candidate["rerank_score"] = rerank_score(new_employee, candidate)

        ranked_candidates = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)
        assigned_buddy = ranked_candidates[0]
        assigned_buddy["llm_reason"] = self.explain_match(new_employee, assigned_buddy, ranked_candidates[:3])
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


def assign_command(args: argparse.Namespace) -> None:
    matcher = BuddyMatcher()
    buddy = matcher.assign_buddy(read_json(args.new_employee))
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Buddy assignment POC using Gemini embeddings, Pinecone, and RAG.")
    subparsers = parser.add_subparsers(required=True)

    seed_parser = subparsers.add_parser("seed", help="Embed and upsert existing employee profiles.")
    seed_parser.add_argument("--employees", default="employees.json")
    seed_parser.set_defaults(func=seed_command)

    assign_parser = subparsers.add_parser("assign", help="Assign a buddy for a new employee profile.")
    assign_parser.add_argument("--new-employee", default="new_employee.json")
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
