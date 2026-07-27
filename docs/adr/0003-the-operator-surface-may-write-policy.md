# The Operator Surface may write Policy

ADR 0001 gave Policy exactly one writer and named that writer "a human".
Approving a Candidate or withholding a model then meant editing YAML by
hand, which is friction the tool can remove. So Policy's writer is now
the operator, acting either through an editor or through the Operator
Surface. The invariant ADR 0001 actually protects survives intact: no
part of the run path — `probe`, `reduce`, `plan`, `generate`, `run`,
`watch`, the scheduled tick — writes Policy, so the tools still cannot
fight the human.

## Why the narrowing is safe

ADR 0001 rejected a single file for a specific reason: "Re-enable a
provider by hand and the next quota check disables it again, or worse,
overwrites the edit mid-save." Both halves of that fear are about an
unattended process. An Operator Surface write is neither unattended nor
concurrent with itself. It happens because the operator asked for it,
once, at a keyboard or through an agent acting for them.

The remaining risk is different, and smaller: the Surface can race the
operator's own editor. A write therefore takes the lock from ADR 0002,
re-reads Policy, and refuses when the file changed since it was read.
It prints the change it applied, so a surprising result is always
traceable to a line.

## Considered options

**Emit patches, never write.** Each command prints the YAML fragment to
add and the operator applies it. Keeps ADR 0001 word for word.
Rejected: it leaves the friction in place, and an agent then needs its
own edit step, which is a second unreviewed writer by another name.

**Write only reversible fields.** Permit `approved_candidates`,
`withheld`, `alias_overrides` and `pacing`; refuse `providers`,
`declared`, `translation_overrides` and `proxy_settings`. Rejected: the
permission table is a second copy of the schema, and it goes stale
against the schema with no symptom — the failure this repo already
documents for hand-maintained lists.

**An overlay file Policy composes.** Preserves the old rule literally.
Rejected: it adds a merge order and two places to look, so every
surprising result starts with "which file won".

## Consequences

Policy is no longer safe to assume hand-written. Anything that reports
Policy's provenance must say which path wrote a line, which is why the
Surface prints its diff.

`validate` becomes load-bearing for a second reason. It was a check the
operator ran; it is now also the gate the Surface applies to its own
output before it promotes a write.
