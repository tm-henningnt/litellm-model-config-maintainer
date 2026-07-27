"""Tests for `litellm_maintainer.operator_surface`.

Every test copies `policy.example.yaml` into `tmp_path` and edits that
copy, never the repository's own file. Test names use the glossary
vocabulary from CONTEXT.md: Policy, Candidate, Withheld, Alias and
Entitlement are precise terms, not synonyms for each other.

The point of this suite is requirement 1 of the ticket: an Operator
Surface write must never destroy a comment or reorder a key. See
`test_every_comment_survives_a_write` and
`test_key_order_is_unchanged_after_a_write`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from litellm_maintainer import operator_surface as opsurf
from litellm_maintainer.policy import PolicyError, parse_policy

EXAMPLE_POLICY = Path(__file__).parent.parent / "policy.example.yaml"

# A comment string from policy.example.yaml, copied verbatim. If this
# string is not byte-for-byte present after a write, the write
# destroyed a comment.
DISTINCTIVE_COMMENT = (
    "# account entitlement. An unpriced Offering carries the pricing kind"
)

EXISTING_CANDIDATE = "example-free-mirror:vendor/new-model:free"
EXISTING_WITHHELD_ID = "example-mixed-provider:coder-experimental"
EXISTING_WITHHELD_REASON = "billing terms unclear, clear this entry once confirmed"
EXISTING_ALIAS_ID = "example-mixed-provider:coder-large"
EXISTING_ALIAS = "claude-mixed-coder-xl"
EXISTING_PROVIDER = "example-subscription-pool"


@pytest.fixture()
def policy_path(tmp_path: Path) -> Path:
    destination = tmp_path / "policy.yaml"
    shutil.copy2(EXAMPLE_POLICY, destination)
    return destination


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return home


def _top_level_keys(text: str) -> list[str]:
    return [line.split(":", 1)[0] for line in text.splitlines() if line[:1].isalpha() and ":" in line]


def _comment_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip().startswith("#")]


# ---------------------------------------------------------------------------
# 1. approve_candidate
# ---------------------------------------------------------------------------


def test_approve_candidate_adds_one_entry_and_still_parses(policy_path: Path, tmp_path: Path):
    new_id = "example-free-mirror:vendor/another-model:free"
    result = opsurf.approve_candidate(policy_path, new_id, home=_home(tmp_path))
    assert result.changed is True
    text = policy_path.read_text()
    policy = parse_policy(yaml.safe_load(text))
    assert new_id in policy.approved_candidates
    assert EXISTING_CANDIDATE in policy.approved_candidates
    assert len(policy.approved_candidates) == 2


# ---------------------------------------------------------------------------
# 2. withhold
# ---------------------------------------------------------------------------


def test_withhold_adds_one_entry_with_reason_and_still_parses(policy_path: Path, tmp_path: Path):
    new_id = "example-mixed-provider:coder-small"
    reason = "quota exhausted for the month"
    result = opsurf.withhold(policy_path, new_id, reason, home=_home(tmp_path))
    assert result.changed is True
    policy = parse_policy(yaml.safe_load(policy_path.read_text()))
    assert policy.withheld[new_id] == reason
    assert policy.withheld[EXISTING_WITHHELD_ID] == EXISTING_WITHHELD_REASON
    assert len(policy.withheld) == 2


# ---------------------------------------------------------------------------
# 3. unwithhold
# ---------------------------------------------------------------------------


def test_unwithhold_removes_exactly_that_entry(policy_path: Path, tmp_path: Path):
    second_id = "example-mixed-provider:coder-small"
    opsurf.withhold(policy_path, second_id, "temporary", home=_home(tmp_path))

    result = opsurf.unwithhold(policy_path, EXISTING_WITHHELD_ID, home=_home(tmp_path))
    assert result.changed is True

    policy = parse_policy(yaml.safe_load(policy_path.read_text()))
    assert EXISTING_WITHHELD_ID not in policy.withheld
    assert policy.withheld[second_id] == "temporary"
    assert len(policy.withheld) == 1


# ---------------------------------------------------------------------------
# 4. set_alias
# ---------------------------------------------------------------------------


def test_set_alias_adds_one_entry_under_naming_alias_overrides(policy_path: Path, tmp_path: Path):
    offering_id = "example-legacy-provider:legacy-model-1"
    result = opsurf.set_alias(policy_path, offering_id, "claude-legacy-one", home=_home(tmp_path))
    assert result.changed is True
    policy = parse_policy(yaml.safe_load(policy_path.read_text()))
    assert policy.naming.alias_overrides[offering_id] == "claude-legacy-one"
    assert policy.naming.alias_overrides[EXISTING_ALIAS_ID] == EXISTING_ALIAS
    assert len(policy.naming.alias_overrides) == 2


# ---------------------------------------------------------------------------
# 5. set_entitlement: sets the value, rejects an unknown one
# ---------------------------------------------------------------------------


def test_set_entitlement_sets_providers_entitlement(policy_path: Path, tmp_path: Path):
    result = opsurf.set_entitlement(
        policy_path, EXISTING_PROVIDER, "shared_pool", home=_home(tmp_path)
    )
    assert result.changed is True
    policy = parse_policy(yaml.safe_load(policy_path.read_text()))
    assert policy.providers[EXISTING_PROVIDER].entitlement == "shared_pool"


def test_set_entitlement_rejects_a_value_outside_the_valid_set(policy_path: Path, tmp_path: Path):
    before = policy_path.read_text()
    with pytest.raises(opsurf.OperatorSurfaceError):
        opsurf.set_entitlement(policy_path, EXISTING_PROVIDER, "bottomless_pool", home=_home(tmp_path))
    assert policy_path.read_text() == before


# ---------------------------------------------------------------------------
# 6. set_entitlement on an unknown provider
# ---------------------------------------------------------------------------


def test_set_entitlement_on_unknown_provider_raises(policy_path: Path, tmp_path: Path):
    before = policy_path.read_text()
    with pytest.raises(opsurf.OperatorSurfaceError):
        opsurf.set_entitlement(policy_path, "no-such-provider", "shared_pool", home=_home(tmp_path))
    assert policy_path.read_text() == before


# ---------------------------------------------------------------------------
# 7. every comment survives -- the point of this ticket
# ---------------------------------------------------------------------------


def test_every_comment_survives_a_write(policy_path: Path, tmp_path: Path):
    before_text = policy_path.read_text()
    before_comment_count = len(_comment_lines(before_text))
    assert DISTINCTIVE_COMMENT in before_text

    home = _home(tmp_path)
    opsurf.approve_candidate(policy_path, "example-free-mirror:vendor/comment-check:free", home=home)
    opsurf.withhold(policy_path, "example-mixed-provider:coder-small", "checking comments", home=home)
    opsurf.set_alias(policy_path, "example-legacy-provider:legacy-model-1", "claude-legacy-check", home=home)
    opsurf.set_entitlement(policy_path, EXISTING_PROVIDER, "shared_pool", home=home)

    after_text = policy_path.read_text()
    after_comment_count = len(_comment_lines(after_text))
    assert after_comment_count == before_comment_count
    assert DISTINCTIVE_COMMENT in after_text
    # Every original comment line, verbatim, is still present.
    assert set(_comment_lines(before_text)) <= set(_comment_lines(after_text))


# ---------------------------------------------------------------------------
# 8. top-level key order is unchanged
# ---------------------------------------------------------------------------


def test_key_order_is_unchanged_after_a_write(policy_path: Path, tmp_path: Path):
    before_keys = _top_level_keys(policy_path.read_text())
    opsurf.approve_candidate(policy_path, "example-free-mirror:vendor/order-check:free", home=_home(tmp_path))
    after_keys = _top_level_keys(policy_path.read_text())
    assert after_keys == before_keys


# ---------------------------------------------------------------------------
# 9. a second identical call is a no-op
# ---------------------------------------------------------------------------


def test_second_identical_call_returns_unchanged_and_writes_nothing(policy_path: Path, tmp_path: Path):
    home = _home(tmp_path)
    new_id = "example-free-mirror:vendor/idempotent-check:free"
    first = opsurf.approve_candidate(policy_path, new_id, home=home)
    assert first.changed is True

    mtime_after_first = policy_path.stat().st_mtime_ns
    text_after_first = policy_path.read_text()

    second = opsurf.approve_candidate(policy_path, new_id, home=home)
    assert second.changed is False
    assert second.diff == ""
    assert policy_path.stat().st_mtime_ns == mtime_after_first
    assert policy_path.read_text() == text_after_first


# ---------------------------------------------------------------------------
# 10. dry_run
# ---------------------------------------------------------------------------


def test_dry_run_returns_a_diff_and_leaves_the_file_untouched(policy_path: Path, tmp_path: Path):
    before_bytes = policy_path.read_bytes()
    result = opsurf.withhold(
        policy_path, "example-mixed-provider:coder-small", "dry run only", home=_home(tmp_path), dry_run=True
    )
    assert result.changed is True
    assert result.diff != ""
    assert policy_path.read_bytes() == before_bytes


# ---------------------------------------------------------------------------
# 11. a write that would produce an invalid Policy is refused
# ---------------------------------------------------------------------------


def test_invalid_result_is_refused_and_file_is_untouched(policy_path: Path, tmp_path: Path):
    before_bytes = policy_path.read_bytes()
    offering_id = "example-legacy-provider:legacy-model-1"
    # An empty Alias fails policy.parse_policy's `_require_str` check on
    # naming.alias_overrides.<id>, a genuine PolicyError, not a fake one.
    with pytest.raises(opsurf.OperatorSurfaceError):
        opsurf.set_alias(policy_path, offering_id, "", home=_home(tmp_path))
    assert policy_path.read_bytes() == before_bytes
    # Prove the premise: parse_policy really does reject this shape.
    raw = yaml.safe_load(before_bytes.decode())
    raw.setdefault("naming", {}).setdefault("alias_overrides", {})[offering_id] = ""
    with pytest.raises(PolicyError):
        parse_policy(raw)


# ---------------------------------------------------------------------------
# 12. a concurrent modification between read and write is refused
# ---------------------------------------------------------------------------


def test_concurrent_modification_is_refused(policy_path: Path, tmp_path: Path, monkeypatch):
    def mutate_file_mid_operation() -> None:
        with open(policy_path, "a") as f:
            f.write("\n# mutated by a concurrent editor\n")

    monkeypatch.setattr(opsurf, "_simulate_race_point", mutate_file_mid_operation)

    before_bytes = policy_path.read_bytes()
    with pytest.raises(opsurf.OperatorSurfaceError) as excinfo:
        opsurf.approve_candidate(
            policy_path, "example-free-mirror:vendor/race-check:free", home=_home(tmp_path)
        )
    assert "changed on disk" in str(excinfo.value)
    assert "no write happened" in str(excinfo.value).lower()
    # The concurrent mutation itself is the only change; our own write
    # must not have landed on top of it.
    assert policy_path.read_bytes() == before_bytes + b"\n# mutated by a concurrent editor\n"


# ---------------------------------------------------------------------------
# 13. no temporary file survives, on success or refusal
# ---------------------------------------------------------------------------


def _stray_temp_files(directory: Path) -> list[Path]:
    return [p for p in directory.iterdir() if p.name.startswith(".") and p.name.endswith(".tmp")]


def test_no_temporary_file_remains_after_success(policy_path: Path, tmp_path: Path):
    opsurf.approve_candidate(
        policy_path, "example-free-mirror:vendor/tmp-check-ok:free", home=_home(tmp_path)
    )
    assert _stray_temp_files(policy_path.parent) == []


def test_no_temporary_file_remains_after_a_refusal(policy_path: Path, tmp_path: Path):
    with pytest.raises(opsurf.OperatorSurfaceError):
        opsurf.set_entitlement(policy_path, "no-such-provider", "shared_pool", home=_home(tmp_path))
    assert _stray_temp_files(policy_path.parent) == []


def test_no_temporary_file_remains_after_invalid_result_refusal(policy_path: Path, tmp_path: Path):
    with pytest.raises(opsurf.OperatorSurfaceError):
        opsurf.set_alias(
            policy_path, "example-legacy-provider:legacy-model-1", "", home=_home(tmp_path)
        )
    assert _stray_temp_files(policy_path.parent) == []


# ---------------------------------------------------------------------------
# 14. the diff names the added or removed line
# ---------------------------------------------------------------------------


def test_diff_names_the_added_line(policy_path: Path, tmp_path: Path):
    new_id = "example-free-mirror:vendor/diff-check:free"
    result = opsurf.approve_candidate(policy_path, new_id, home=_home(tmp_path), dry_run=True)
    added_lines = [line for line in result.diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    assert any(new_id in line for line in added_lines)


def test_diff_names_the_removed_line(policy_path: Path, tmp_path: Path):
    result = opsurf.unwithhold(policy_path, EXISTING_WITHHELD_ID, home=_home(tmp_path), dry_run=True)
    removed_lines = [line for line in result.diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    assert any(EXISTING_WITHHELD_ID in line for line in removed_lines)
