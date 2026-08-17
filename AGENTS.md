# AGENTS.md

Conventions for agent tools working in this repository.

## Lint

```
ruff check .
```

The rules are stated in `pyproject.toml`, never inherited from the
installed ruff's defaults. Ruff 0.15.12 and 0.16.0 disagree by 219
findings on this tree, so an unstated rule set gates nothing.

Selected: `E`, `F`, `W`. Keep this at zero findings. Adopting another rule
family is a deliberate sweep — say so in the commit, and do it on its own.

## Agent skills

### Issue tracker

Local markdown under `.scratch/`, which is gitignored because this repo
is public. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles with their default strings, recorded as a
`Status:` line in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See
`docs/agents/domain.md`.
