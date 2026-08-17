# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps
those roles to the strings this repo uses.

This repo tracks issues as local markdown files, so a role is not a
tracker label. Write it as the `Status:` line near the top of the issue
file. See `issue-tracker.md`.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill names a role, for example "apply the AFK-ready triage
label", use the string from the right-hand column.

Edit the right-hand column to match a different vocabulary.
