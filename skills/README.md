# Runtime Skills

This directory stores the repo-local runtime skills for the Day 5 multi-agent
pipeline.

## Layout

- `roles/`: base role prompts and execution parameters
- `domains/`: domain-specific expert guidance, source preferences, and glossaries
- `shared/`: reusable specialist skills injected only when the active role needs them

## Runtime Behavior

`academic_researcher.skills.manager.SkillManager` loads these JSON files just in
time. Only the active role, the resolved domain, and a small set of relevant
glossary entries are injected into the prompt context window.
