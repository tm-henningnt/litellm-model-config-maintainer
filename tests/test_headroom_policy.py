"""Policy's `headroom` block: which Allowance each codexbar Reading joins.

Assert external behaviour: what `parse_policy` returns, and what makes it
raise `PolicyError` naming the offending key. See ADR 0012 and the
headroom spec, decision 3 and decision 5.
"""

from __future__ import annotations

import pytest

from litellm_maintainer.policy import (
    DEFAULT_HEADROOM_COMMAND,
    DEFAULT_HEADROOM_INTERVAL_MINUTES,
    DEFAULT_HEADROOM_TIMEOUT_SECONDS,
    Headroom,
    PolicyError,
    parse_policy,
)
from test_policy import _valid_policy_dict


def test_a_policy_naming_no_headroom_block_gets_the_all_empty_default():
    policy = parse_policy(_valid_policy_dict())

    assert policy.headroom == Headroom()
    assert policy.headroom.sources == {}
    assert policy.headroom.command == DEFAULT_HEADROOM_COMMAND
    assert policy.headroom.interval_minutes == DEFAULT_HEADROOM_INTERVAL_MINUTES
    assert policy.headroom.timeout_seconds == DEFAULT_HEADROOM_TIMEOUT_SECONDS


def test_sources_join_an_allowance_id_to_a_codexbar_identity():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "pool:claude-subscription": "codexbar:claude/operator@example.com",
            "provider:cline": "codexbar:clinepass/operator@example.com",
            "provider:opencode-go": "codexbar:opencodego/",
        }
    }

    policy = parse_policy(raw)

    assert policy.headroom.sources["pool:claude-subscription"] == (
        "codexbar:claude/operator@example.com"
    )
    assert policy.headroom.sources["provider:opencode-go"] == "codexbar:opencodego/"


def test_command_and_interval_minutes_are_optional_with_stated_defaults():
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {}}

    policy = parse_policy(raw)

    assert policy.headroom.command == "codexbar"
    assert policy.headroom.interval_minutes == 15


def test_a_test_may_point_command_at_a_fixture_script():
    raw = _valid_policy_dict()
    raw["headroom"] = {"command": "/path/to/fixture-codexbar", "sources": {}}

    policy = parse_policy(raw)

    assert policy.headroom.command == "/path/to/fixture-codexbar"


@pytest.mark.parametrize(
    "allowance_id",
    ["cline-pass", "claude-subscription", "openai:cline", ""],
)
def test_a_source_key_that_is_not_a_well_formed_allowance_id_is_rejected(allowance_id):
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {allowance_id: "codexbar:claude/operator@example.com"}}

    with pytest.raises(PolicyError, match="headroom.sources"):
        parse_policy(raw)


@pytest.mark.parametrize(
    "source",
    [
        "clinepass/operator@example.com",  # missing the 'codexbar:' prefix
        "codexbar:clinepass",  # missing the '/'
        "codexbar:",
    ],
)
def test_a_malformed_source_value_is_rejected(source):
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {"provider:cline": source}}

    with pytest.raises(PolicyError, match="headroom.sources"):
        parse_policy(raw)


# --- Defect 6: `timeout_seconds` --------------------------------------------
#
# Measured 2026-07-28: 24s for four mapped providers, 21-31s for every
# provider codexbar knows. A fifth or sixth mapped provider plausibly
# crosses the fixed 40s timeout, after which every refresh times out and
# the capability goes stale for good. Stating it in Policy lets the
# operator raise it without a code change.


def test_timeout_seconds_defaults_when_the_headroom_block_names_none():
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {}}

    policy = parse_policy(raw)

    assert policy.headroom.timeout_seconds == DEFAULT_HEADROOM_TIMEOUT_SECONDS


def test_timeout_seconds_may_be_stated_explicitly():
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {}, "timeout_seconds": 90}

    policy = parse_policy(raw)

    assert policy.headroom.timeout_seconds == 90


def test_timeout_seconds_must_be_a_positive_number():
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {}, "timeout_seconds": 0}

    with pytest.raises(PolicyError, match="headroom.timeout_seconds"):
        parse_policy(raw)


def test_timeout_seconds_rejects_a_non_number():
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {}, "timeout_seconds": "40"}

    with pytest.raises(PolicyError, match="headroom.timeout_seconds"):
        parse_policy(raw)


def test_an_unrecognised_headroom_key_is_rejected():
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {}, "not_a_real_key": True}

    with pytest.raises(PolicyError, match="headroom"):
        parse_policy(raw)


def test_the_example_policy_declares_headroom_and_maps_two_codex_seats():
    """Ticket 11: `codexbar --provider codex --all-accounts` returns both
    ChatGPT seats, so the example now maps both instead of leaving codex
    unmapped."""
    from pathlib import Path

    from litellm_maintainer.policy import load_policy

    example = Path(__file__).parent.parent / "policy.example.yaml"
    policy = load_policy(example)

    assert policy.headroom.sources  # the example demonstrates the block, not an empty one
    assert policy.headroom.all_accounts_providers == ("codex",)
    codex_sources = [
        source for source in policy.headroom.sources.values() if source.startswith("codexbar:codex/")
    ]
    assert len(codex_sources) == 2
    assert "codexbar:codex/one@example.com" in codex_sources
    assert "codexbar:codex/two@example.com" in codex_sources


# --- `demote_at_full` (ticket 08) -------------------------------------------
#
# The demotion capability itself lives in `guidance.py` and is tested there.
# This file covers only how Policy parses the flag: default off, explicit
# either way, and that the shipped example states it off.


# --- Ticket 09: a source entry may declare what each slot measures --------
#
# A plain string still means "these three slots are nested time windows",
# byte-identical to before this ticket. A mapping instead names, under
# `windows`, an operator-chosen Sub-allowance id for `primary`, `secondary`
# or `tertiary` -- for a provider like Gemini whose slots hold one quota
# per MODEL. See CONTEXT.md, "Sub-allowance", and `docs/gotchas.md`,
# "codexbar's three window slots do not mean one thing".


def test_a_plain_string_source_declares_no_windows():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {"pool:claude-subscription": "codexbar:claude/operator@example.com"}
    }

    policy = parse_policy(raw)

    assert policy.headroom.sources["pool:claude-subscription"] == (
        "codexbar:claude/operator@example.com"
    )
    assert policy.headroom.source_windows == {}


def test_a_mapping_source_may_name_every_slot():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {
                "source": "codexbar:gemini/operator@example.com",
                "windows": {
                    "primary": "gemini-pro",
                    "secondary": "gemini-flash",
                    "tertiary": "gemini-flash-lite",
                },
            }
        }
    }

    policy = parse_policy(raw)

    assert policy.headroom.sources["provider:gemini"] == "codexbar:gemini/operator@example.com"
    assert policy.headroom.source_windows["provider:gemini"] == {
        "primary": "gemini-pro",
        "secondary": "gemini-flash",
        "tertiary": "gemini-flash-lite",
    }


def test_a_mapping_source_may_name_only_one_slot():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {
                "source": "codexbar:gemini/operator@example.com",
                "windows": {"primary": "gemini-pro"},
            }
        }
    }

    policy = parse_policy(raw)

    assert policy.headroom.source_windows["provider:gemini"] == {"primary": "gemini-pro"}


def test_a_mapping_source_with_no_windows_key_declares_none():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {"source": "codexbar:gemini/operator@example.com"}
        }
    }

    policy = parse_policy(raw)

    assert policy.headroom.source_windows == {}


def test_a_mapping_source_requires_a_source_key():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {"provider:gemini": {"windows": {"primary": "gemini-pro"}}}
    }

    with pytest.raises(PolicyError, match="headroom.sources.provider:gemini.source"):
        parse_policy(raw)


def test_a_mapping_source_rejects_an_unrecognised_key():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {
                "source": "codexbar:gemini/operator@example.com",
                "not_a_real_key": True,
            }
        }
    }

    with pytest.raises(PolicyError, match="headroom.sources.provider:gemini"):
        parse_policy(raw)


def test_a_windows_mapping_rejects_a_slot_name_other_than_the_three():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {
                "source": "codexbar:gemini/operator@example.com",
                "windows": {"quaternary": "gemini-ultra"},
            }
        }
    }

    with pytest.raises(PolicyError, match="headroom.sources.provider:gemini.windows"):
        parse_policy(raw)


def test_a_source_entry_that_is_neither_a_string_nor_a_mapping_is_rejected():
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {"provider:gemini": 7}}

    with pytest.raises(PolicyError, match="headroom.sources.provider:gemini"):
        parse_policy(raw)


def test_a_mapping_sources_own_malformed_source_value_is_still_rejected():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {"source": "gemini/operator@example.com"}  # no 'codexbar:'
        }
    }

    with pytest.raises(PolicyError, match="headroom.sources"):
        parse_policy(raw)


def test_the_example_policy_maps_gemini_with_the_windows_mapping():
    from pathlib import Path

    from litellm_maintainer.policy import load_policy

    example = Path(__file__).parent.parent / "policy.example.yaml"
    policy = load_policy(example)

    gemini_windows = policy.headroom.source_windows["pool:example-gemini"]
    assert gemini_windows == {
        "primary": "gemini-pro",
        "secondary": "gemini-flash",
        "tertiary": "gemini-flash-lite",
    }


# --- Ticket 10: `members` says which Health Key draws on each slot --------
#
# `members` is nested inside a mapping-form source entry, keyed by a slot
# id `windows` already declares in the SAME entry. A member is a Health
# Key: a Feed Offering's own id, or a Declared Offering's Alias -- see
# CONTEXT.md, "Health Key".


def test_a_mapping_source_may_name_members():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {
                "source": "codexbar:gemini/operator@example.com",
                "windows": {
                    "primary": "gemini-pro",
                    "secondary": "gemini-flash",
                    "tertiary": "gemini-flash-lite",
                },
                "members": {
                    "gemini-flash": [
                        "gemini:gemini-3-flash-preview",
                        "gemini:gemini-3.5-flash",
                    ],
                },
            }
        }
    }

    policy = parse_policy(raw)

    assert policy.headroom.source_members["provider:gemini"] == {
        "gemini-flash": ("gemini:gemini-3-flash-preview", "gemini:gemini-3.5-flash"),
    }


def test_a_member_may_name_a_declared_offerings_alias():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {
                "source": "codexbar:gemini/operator@example.com",
                "windows": {"primary": "gemini-pro"},
                "members": {"gemini-pro": ["claude-fable-5"]},
            }
        }
    }

    policy = parse_policy(raw)

    assert policy.headroom.source_members["provider:gemini"]["gemini-pro"] == ("claude-fable-5",)


def test_no_members_key_declares_none():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {
                "source": "codexbar:gemini/operator@example.com",
                "windows": {"primary": "gemini-pro"},
            }
        }
    }

    policy = parse_policy(raw)

    assert policy.headroom.source_members == {}


def test_a_plain_string_source_declares_no_members():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {"pool:claude-subscription": "codexbar:claude/operator@example.com"}
    }

    policy = parse_policy(raw)

    assert policy.headroom.source_members == {}


def test_a_member_key_naming_no_declared_slot_still_parses():
    """A `members` key may name a codexbar extra window instead of a slot.

    Claude's `claude-weekly-scoped-fable` is an `extraRateWindows` entry,
    never one of the three named slots, so a rule that demanded a declared
    slot refused the only honest way to state it. `doctor` reports a key
    that reaches neither.
    """
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "pool:claude-subscription": {
                "source": "codexbar:claude/",
                "members": {"claude-weekly-scoped-fable": ["claude-fable-5"]},
            }
        }
    }

    policy = parse_policy(raw)

    members = policy.headroom.source_members["pool:claude-subscription"]
    assert members["claude-weekly-scoped-fable"] == ("claude-fable-5",)


def test_members_with_no_windows_at_all_still_parses():
    """Policy never fails to parse over a member key.

    Measured 2026-07-28: codexbar published
    `claude-weekly-scoped-all-model` at 18:48Z and had dropped it by
    20:52Z. A Policy naming a window the vendor retires must not stop the
    Generator for every provider.
    """
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {
                "source": "codexbar:gemini/operator@example.com",
                "members": {"gemini-pro": ["gemini:gemini-3-pro-preview"]},
            }
        }
    }

    policy = parse_policy(raw)

    assert policy.headroom.source_members["provider:gemini"]["gemini-pro"] == (
        "gemini:gemini-3-pro-preview",
    )


def test_a_members_value_that_is_not_a_list_is_rejected():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {
                "source": "codexbar:gemini/operator@example.com",
                "windows": {"primary": "gemini-pro"},
                "members": {"gemini-pro": "gemini:gemini-3-pro-preview"},
            }
        }
    }

    with pytest.raises(PolicyError, match="headroom.sources.provider:gemini.members.gemini-pro"):
        parse_policy(raw)


def test_a_member_naming_no_known_health_key_still_parses():
    # A typo, or a model the Feed dropped: Policy parsing never fails on
    # this. `doctor` reports the gap (ticket 10).
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {
                "source": "codexbar:gemini/operator@example.com",
                "windows": {"primary": "gemini-pro"},
                "members": {"gemini-pro": ["gemini:no-such-offering"]},
            }
        }
    }

    policy = parse_policy(raw)

    assert policy.headroom.source_members["provider:gemini"]["gemini-pro"] == (
        "gemini:no-such-offering",
    )


def test_the_example_policy_declares_members_for_every_gemini_slot():
    from pathlib import Path

    from litellm_maintainer.policy import load_policy

    example = Path(__file__).parent.parent / "policy.example.yaml"
    policy = load_policy(example)

    members = policy.headroom.source_members["pool:example-gemini"]
    assert members == {
        "gemini-pro": ("claude-example-gemini-pro",),
        "gemini-flash": ("claude-example-gemini-flash",),
        "gemini-flash-lite": ("claude-example-gemini-flash-lite",),
    }


def test_demote_at_full_defaults_false_when_the_headroom_block_names_no_flag():
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {}}

    policy = parse_policy(raw)

    assert policy.headroom.demote_at_full is False


def test_demote_at_full_defaults_false_with_no_headroom_block_at_all():
    policy = parse_policy(_valid_policy_dict())

    assert policy.headroom.demote_at_full is False


def test_demote_at_full_may_be_stated_true():
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {}, "demote_at_full": True}

    policy = parse_policy(raw)

    assert policy.headroom.demote_at_full is True


def test_demote_at_full_must_be_a_bool():
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {}, "demote_at_full": "true"}

    with pytest.raises(PolicyError, match="headroom.demote_at_full"):
        parse_policy(raw)


# --- Ticket 11: `all_accounts_providers` -----------------------------------
#
# `codexbar --provider codex --all-accounts` returns two Readings sharing
# one `providerID`. `all_accounts_providers` names, once, which codexbar
# provider ids need that call instead of the batched one.


def test_all_accounts_providers_defaults_empty():
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {}}

    policy = parse_policy(raw)

    assert policy.headroom.all_accounts_providers == ()


def test_all_accounts_providers_defaults_empty_with_no_headroom_block_at_all():
    policy = parse_policy(_valid_policy_dict())

    assert policy.headroom.all_accounts_providers == ()


def test_all_accounts_providers_may_be_stated():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "credential:SEAT1_KEY": "codexbar:codex/one@example.com",
            "credential:SEAT2_KEY": "codexbar:codex/two@example.com",
        },
        "all_accounts_providers": ["codex"],
    }

    policy = parse_policy(raw)

    assert policy.headroom.all_accounts_providers == ("codex",)


def test_all_accounts_providers_must_be_a_list():
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {}, "all_accounts_providers": "codex"}

    with pytest.raises(PolicyError, match="headroom.all_accounts_providers"):
        parse_policy(raw)


def test_all_accounts_providers_rejects_a_non_string_item():
    raw = _valid_policy_dict()
    raw["headroom"] = {"sources": {}, "all_accounts_providers": [7]}

    with pytest.raises(PolicyError, match="headroom.all_accounts_providers"):
        parse_policy(raw)


def test_the_example_policy_ships_with_demote_at_full_off():
    from pathlib import Path

    from litellm_maintainer.policy import load_policy

    example = Path(__file__).parent.parent / "policy.example.yaml"
    policy = load_policy(example)

    assert policy.headroom.demote_at_full is False


# --- `unmeasured`: a Health Key that draws on no published window --------
#
# Gemini's three slots hold Pro, Flash and Flash Lite. The same account also
# serves Gemma, which none of the three measures. Before this key the
# operator could only leave Gemma out (and fail `member.unclaimed`) or file
# it under a slot that does not measure it (and state something false).


def test_a_mapping_source_may_name_unmeasured_health_keys():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {
                "source": "codexbar:gemini/operator@example.com",
                "windows": {"primary": "gemini-pro"},
                "members": {"gemini-pro": ["gemini:gemini-3-pro"]},
                "unmeasured": ["gemini:gemma-4-31b-it"],
            }
        }
    }

    policy = parse_policy(raw)

    assert policy.headroom.source_unmeasured == {
        "provider:gemini": ("gemini:gemma-4-31b-it",)
    }


def test_no_unmeasured_key_declares_none():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {"pool:claude-subscription": "codexbar:claude/operator@example.com"}
    }

    policy = parse_policy(raw)

    assert policy.headroom.source_unmeasured == {}


def test_a_health_key_both_measured_and_unmeasured_is_rejected():
    # A Health Key draws on one window or on none. Stating both leaves the
    # reader no way to know which the operator meant.
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {
                "source": "codexbar:gemini/operator@example.com",
                "windows": {"primary": "gemini-pro"},
                "members": {"gemini-pro": ["gemini:gemini-3-pro"]},
                "unmeasured": ["gemini:gemini-3-pro"],
            }
        }
    }

    with pytest.raises(PolicyError, match="draws on one window or on none"):
        parse_policy(raw)


def test_an_unmeasured_value_that_is_not_a_list_is_rejected():
    raw = _valid_policy_dict()
    raw["headroom"] = {
        "sources": {
            "provider:gemini": {
                "source": "codexbar:gemini/operator@example.com",
                "unmeasured": "gemini:gemma-4-31b-it",
            }
        }
    }

    with pytest.raises(PolicyError):
        parse_policy(raw)
