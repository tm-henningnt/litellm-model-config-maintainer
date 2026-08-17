"""`deploy` applies a change now, and says what that costs.

The scheduled tick is the only ordinary writer, and its interval is an
hour. `run --force` does not shorten the wait: `schedule.due` takes no
force parameter, so the gate never sees the flag.

`deploy` writes whatever the interval says. It resolves the file the
proxy reads from the installed tick job, because passing the wrong
`--out` writes the right config where nothing reads it.
"""

from __future__ import annotations

import plistlib

from litellm_maintainer.cli import installed_out_path


def _write_plist(target_dir, arguments):
    from litellm_maintainer.schedule import DEFAULT_LABEL, plist_path

    target_dir.mkdir(parents=True, exist_ok=True)
    path = plist_path(target_dir, DEFAULT_LABEL)
    path.write_bytes(
        plistlib.dumps({"Label": DEFAULT_LABEL, "ProgramArguments": arguments})
    )
    return path


def test_the_out_path_comes_from_the_installed_tick_job(tmp_path):
    """The plist is the only record of which file the proxy reads."""
    agents = tmp_path / "LaunchAgents"
    _write_plist(
        agents,
        ["python", "-m", "litellm_maintainer.tick_entry", "run", "--out", "/srv/config.yaml"],
    )

    resolved, source = installed_out_path(agents)

    assert str(resolved) == "/srv/config.yaml"
    assert "no.tallmaker" in source


def test_no_installed_job_resolves_nothing_rather_than_guessing(tmp_path):
    """A guess writes the right config to the wrong file, which looks
    like a deploy that did nothing."""
    agents = tmp_path / "LaunchAgents"
    agents.mkdir()

    resolved, why = installed_out_path(agents)

    assert resolved is None
    assert "installed" in why


def test_a_job_that_passes_no_out_resolves_nothing(tmp_path):
    agents = tmp_path / "LaunchAgents"
    _write_plist(agents, ["python", "-m", "litellm_maintainer.tick_entry", "run"])

    resolved, why = installed_out_path(agents)

    assert resolved is None
    assert "--out" in why


def test_an_unreadable_plist_resolves_nothing(tmp_path):
    from litellm_maintainer.schedule import DEFAULT_LABEL, plist_path

    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    plist_path(agents, DEFAULT_LABEL).write_text("not a plist")

    resolved, why = installed_out_path(agents)

    assert resolved is None
    assert "could not be read" in why


def test_a_missing_launch_agents_directory_resolves_nothing(tmp_path):
    resolved, why = installed_out_path(tmp_path / "absent")

    assert resolved is None
    assert "LaunchAgents" in why


def test_deploy_refuses_when_it_cannot_learn_where_the_proxy_reads(tmp_path, capsys):
    """Refusing beats writing somewhere harmless-looking."""
    import argparse

    from litellm_maintainer.cli import cmd_deploy

    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    args = argparse.Namespace(
        out=None,
        feed=None,
        policy=None,
        home=str(tmp_path / "home"),
        env=None,
        force=False,
        target_dir=agents,
    )

    exit_code = cmd_deploy(args)

    assert exit_code == 1
    assert "Refused to deploy" in capsys.readouterr().err
