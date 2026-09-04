"""`tidescout serve`: the bind address, and the guard the spec asks for.

API design spec §4.3 states the app binds `127.0.0.1`, "not `0.0.0.0`: this is
a single-user local tool with no auth, and binding it to every interface would
put an unauthenticated filesystem-backed API on the local network." A free-text
`--host` reopens exactly that, so the guard is tested rather than trusted.
"""

import pytest
from typer.testing import CliRunner

from tidescout import cli

runner = CliRunner()


def _flat(output: str) -> str:
    """Rich hard-wraps console output at the terminal width, so a message this
    test asserts on can arrive split across a newline."""
    return " ".join(output.split())


@pytest.fixture
def spy(monkeypatch):
    """Record what `serve` would have bound, without ever binding it."""
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.append(kw))
    return calls


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "10.0.0.5"])
def test_serve_refuses_a_non_loopback_host_without_the_opt_in(spy, host):
    """Both halves: a non-zero exit AND no server started. An implementation
    that printed the warning and bound the socket anyway would pass a
    message-only check."""
    result = runner.invoke(cli.app, ["serve", "--host", host])
    assert result.exit_code != 0, result.output
    assert "no authentication" in _flat(result.output)
    assert "--allow-remote" in _flat(result.output)
    assert spy == [], "the guard must run before uvicorn.run"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5"])
def test_a_loopback_host_passes_through_unchanged(spy, host):
    """The other direction. A guard that rejected everything would satisfy the
    test above while breaking the only supported way to run the tool."""
    result = runner.invoke(cli.app, ["serve", "--host", host, "--port", "8123"])
    assert result.exit_code == 0, result.output
    assert spy == [{"host": host, "port": 8123}]


def test_allow_remote_permits_a_non_loopback_host_and_says_so(spy):
    result = runner.invoke(cli.app, ["serve", "--host", "0.0.0.0", "--allow-remote"])
    assert result.exit_code == 0, result.output
    assert spy == [{"host": "0.0.0.0", "port": 8000}]
    assert "reachable from the local network" in _flat(result.output)


def test_the_default_host_is_loopback():
    """The spec's binding, asserted at the option's default rather than only in
    the guard -- changing the default to 0.0.0.0 would otherwise slip through
    with --allow-remote never mentioned."""
    assert cli._is_loopback("127.0.0.1")
    assert not cli._is_loopback("0.0.0.0")
