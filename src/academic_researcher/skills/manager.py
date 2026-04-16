from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _normalize_label(value: str) -> str:
    return "".join(ch for ch in value.lower().strip() if ch.isalnum())


def _tokenize(value: str) -> set[str]:
    token = []
    tokens: set[str] = set()
    for ch in value.lower():
        if ch.isalnum():
            token.append(ch)
            continue
        if token:
            tokens.add("".join(token))
            token = []
    if token:
        tokens.add("".join(token))
    return tokens


@dataclass(frozen=True)
class SharedSkill:
    name: str
    display_name: str
    description: str
    instruction: str
    guidelines: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RoleSkill:
    name: str
    display_name: str
    description: str
    system_prompt: str
    responsibilities: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    shared_skills: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GlossaryTerm:
    term: str
    definition: str
    aliases: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DomainSkill:
    id: str
    display_name: str
    description: str
    aliases: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    expert_prompt: str = ""
    source_preferences: List[str] = field(default_factory=list)
    role_guidance: Dict[str, str] = field(default_factory=dict)
    glossary: List[GlossaryTerm] = field(default_factory=list)


class SkillManager:
    """Load role, shared, and domain skill configs just in time."""

    def __init__(self, skills_root: Optional[str | Path] = None):
        configured_root = skills_root or os.getenv("ACADEMIC_RESEARCHER_SKILLS_DIR")
        self.skills_root = Path(configured_root) if configured_root else self._default_root()

    @staticmethod
    def _default_root() -> Path:
        return Path(__file__).resolve().parents[3] / "skills"

    def resolve_domain(
        self,
        *,
        topic: str,
        goal: str = "",
        explicit_domain: Optional[str] = None,
    ) -> str:
        if explicit_domain:
            explicit = self._match_domain(explicit_domain)
            if explicit:
                return explicit.id

        combined = f"{topic} {goal}".strip()
        best_id = "general"
        best_score = 0

        for domain in self.list_domains():
            if domain.id == "general":
                continue
            score = self._score_domain_match(domain, combined)
            if score > best_score:
                best_score = score
                best_id = domain.id

        return best_id

    def build_system_prompt(
        self,
        role: str,
        *,
        topic: str,
        goal: str,
        domain: Optional[str] = None,
    ) -> str:
        role_skill = self.load_role(role)
        domain_skill = self.load_domain(domain or "general")

        sections = [role_skill.system_prompt]
        if role_skill.responsibilities:
            sections.append("Responsibilities:")
            sections.extend(f"- {item}" for item in role_skill.responsibilities)
        if role_skill.constraints:
            sections.append("Constraints:")
            sections.extend(f"- {item}" for item in role_skill.constraints)

        sections.append(
            f"Active domain: {domain_skill.display_name}. {domain_skill.description}"
        )
        if domain_skill.expert_prompt:
            sections.append(f"Domain expertise: {domain_skill.expert_prompt}")

        role_guidance = domain_skill.role_guidance.get(role)
        if role_guidance:
            sections.append(f"Role-specific domain guidance: {role_guidance}")

        if role == "planner":
            min_queries = role_skill.parameters.get("search_query_min")
            max_queries = role_skill.parameters.get("search_query_max")
            if min_queries and max_queries:
                sections.append(
                    f"Return between {min_queries} and {max_queries} targeted search queries."
                )

        sections.append(
            "Keep output directly useful for the next agent and avoid redundant exposition."
        )
        return "\n".join(sections)

    def build_runtime_context(
        self,
        role: str,
        *,
        topic: str,
        goal: str,
        domain: Optional[str] = None,
        additional_text: str = "",
        user_context: str = "",
    ) -> str:
        role_skill = self.load_role(role)
        domain_skill = self.load_domain(domain or "general")
        glossary_limit = int(role_skill.parameters.get("glossary_term_limit", 4))
        matched_terms = self._select_glossary_terms(
            domain_skill,
            f"{topic}\n{goal}\n{additional_text}\n{user_context}",
            limit=glossary_limit,
        )

        parts = [
            "Skill Context:",
            f"- Active Domain: {domain_skill.display_name}",
        ]

        if domain_skill.source_preferences:
            parts.append("- Source Preferences:")
            parts.extend(f"  - {item}" for item in domain_skill.source_preferences[:2])

        shared = self._load_shared_skills(role_skill.shared_skills)
        if shared:
            parts.append("- Specialist Skills:")
            for item in shared:
                parts.append(f"  - {item.display_name}: {item.instruction}")
                for guideline in item.guidelines[:2]:
                    parts.append(f"    - {guideline}")

        if matched_terms:
            parts.append("- Relevant Terminology:")
            for term in matched_terms:
                parts.append(f"  - {term.term}: {term.definition}")

        if user_context:
            parts.append("- Historical Context:")
            for line in user_context.splitlines():
                parts.append(f"  {line}")

        return "\n".join(parts)

    def get_role_parameter(self, role: str, name: str, default: Any = None) -> Any:
        return self.load_role(role).parameters.get(name, default)

    def list_domains(self) -> List[DomainSkill]:
        domains_dir = self.skills_root / "domains"
        return [self.load_domain(path.stem) for path in sorted(domains_dir.glob("*.json"))]

    def load_role(self, role: str) -> RoleSkill:
        data = self._load_json("roles", role)
        return RoleSkill(
            name=data["name"],
            display_name=data.get("display_name", data["name"].title()),
            description=data.get("description", ""),
            system_prompt=data["system_prompt"],
            responsibilities=list(data.get("responsibilities", [])),
            constraints=list(data.get("constraints", [])),
            shared_skills=list(data.get("shared_skills", [])),
            parameters=dict(data.get("parameters", {})),
        )

    def load_domain(self, domain: str) -> DomainSkill:
        matched = self._match_domain(domain)
        if matched is None:
            if domain != "general":
                return self.load_domain("general")
            raise FileNotFoundError(f"Domain skill {domain!r} not found in {self.skills_root}")
        return matched

    def list_domains_raw(self) -> List[Dict[str, Any]]:
        domains_dir = self.skills_root / "domains"
        return [self._load_json("domains", path.stem) for path in sorted(domains_dir.glob("*.json"))]

    def _match_domain(self, domain: str) -> Optional[DomainSkill]:
        wanted = _normalize_label(domain)
        for candidate in self.list_domains_raw():
            aliases = {_normalize_label(candidate["id"])}
            aliases.update(_normalize_label(alias) for alias in candidate.get("aliases", []))
            if wanted in aliases:
                return self._domain_from_data(candidate)
        return None

    def _domain_from_data(self, data: Dict[str, Any]) -> DomainSkill:
        glossary = [
            GlossaryTerm(
                term=item["term"],
                definition=item["definition"],
                aliases=list(item.get("aliases", [])),
            )
            for item in data.get("glossary", [])
        ]
        return DomainSkill(
            id=data["id"],
            display_name=data.get("display_name", data["id"].replace("_", " ").title()),
            description=data.get("description", ""),
            aliases=list(data.get("aliases", [])),
            keywords=list(data.get("keywords", [])),
            expert_prompt=data.get("expert_prompt", ""),
            source_preferences=list(data.get("source_preferences", [])),
            role_guidance=dict(data.get("role_guidance", {})),
            glossary=glossary,
        )

    def _load_shared_skills(self, names: Iterable[str]) -> List[SharedSkill]:
        items: List[SharedSkill] = []
        for name in names:
            data = self._load_json("shared", name)
            items.append(
                SharedSkill(
                    name=data["name"],
                    display_name=data.get("display_name", data["name"].title()),
                    description=data.get("description", ""),
                    instruction=data.get("instruction", ""),
                    guidelines=list(data.get("guidelines", [])),
                )
            )
        return items

    def _score_domain_match(self, domain: DomainSkill, text: str) -> int:
        normalized_text = _normalize_label(text)
        text_tokens = _tokenize(text)
        score = 0

        aliases = [domain.id, *domain.aliases, *domain.keywords]
        for alias in aliases:
            normalized_alias = _normalize_label(alias)
            if normalized_alias and normalized_alias in normalized_text:
                score += 3
            score += len(_tokenize(alias) & text_tokens)

        for term in domain.glossary:
            for alias in [term.term, *term.aliases]:
                normalized_alias = _normalize_label(alias)
                if normalized_alias and normalized_alias in normalized_text:
                    score += 2
                score += len(_tokenize(alias) & text_tokens)

        return score

    def _select_glossary_terms(
        self,
        domain: DomainSkill,
        text: str,
        *,
        limit: int,
    ) -> List[GlossaryTerm]:
        normalized_text = _normalize_label(text)
        text_tokens = _tokenize(text)
        scored_terms: List[tuple[int, GlossaryTerm]] = []

        for term in domain.glossary:
            score = 0
            for alias in [term.term, *term.aliases]:
                normalized_alias = _normalize_label(alias)
                if normalized_alias and normalized_alias in normalized_text:
                    score += 3
                score += len(_tokenize(alias) & text_tokens)
            if score > 0:
                scored_terms.append((score, term))

        scored_terms.sort(key=lambda item: (-item[0], item[1].term))
        if scored_terms:
            return [term for _, term in scored_terms[:limit]]
        return domain.glossary[: min(limit, 2)]

    @lru_cache(maxsize=64)
    def _load_json(self, section: str, name: str) -> Dict[str, Any]:
        path = self.skills_root / section / f"{name}.json"
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
