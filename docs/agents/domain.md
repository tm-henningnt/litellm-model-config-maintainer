# Domain Docs

How the engineering skills should consume this repo's domain
documentation when exploring the codebase.

This repo is **single-context**: one `CONTEXT.md` and one `docs/adr/` at
the root.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root, or
- **`CONTEXT-MAP.md`** at the repo root if it exists — it points at one
  `CONTEXT.md` per context. Read each one relevant to the topic.
- **`docs/adr/`** — read the ADRs that touch the area you are about to
  work in. In multi-context repos, also check `src/<context>/docs/adr/`
  for context-scoped decisions.

If any of these files do not exist, **proceed silently**. Do not flag
their absence. Do not suggest creating them upfront. The
`/domain-modeling` skill, reached through `/grill-with-docs` and
`/improve-codebase-architecture`, creates them when terms or decisions
actually get resolved.

## File structure

Single-context repo, which is this one:

```
/
├── CONTEXT.md
├── docs/adr/
│   └── 0001-one-writer-per-file.md
├── providers/
└── scripts/
```

Multi-context repo, signalled by a `CONTEXT-MAP.md` at the root:

```
/
├── CONTEXT-MAP.md
├── docs/adr/                          ← system-wide decisions
└── src/
    ├── ordering/
    │   ├── CONTEXT.md
    │   └── docs/adr/                  ← context-specific decisions
    └── billing/
        ├── CONTEXT.md
        └── docs/adr/
```

## Use the glossary's vocabulary

When your output names a domain concept, in an issue title, a refactor
proposal, a hypothesis or a test name, use the term as `CONTEXT.md`
defines it. Do not drift to a synonym the glossary lists under `_Avoid_`.

This matters here. The glossary splits words that look
interchangeable but are not. "Disabled" is three different states:
Withheld, Excluded and Candidate. An Offering is either Discovered or
Declared. Use the precise term.

If the concept you need is not in the glossary, that is a signal. Either
you are inventing language the project does not use, and you should
reconsider, or there is a real gap to note for `/domain-modeling`.

## Flag ADR conflicts

If your output contradicts an ADR, say so instead of overriding it
silently:

> _Contradicts ADR-0001 (one writer per file) — but worth reopening
> because…_
