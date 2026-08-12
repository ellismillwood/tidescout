from typer.testing import CliRunner

import tidescout
from tidescout.cli import app


def test_package_imports():
    assert tidescout.__version__ == "0.1.0"


def test_cli_help_runs():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "TideScout" in result.output
