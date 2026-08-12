import typer

app = typer.Typer(no_args_is_help=True, help="TideScout: SC inshore fishing decision support.")


@app.callback()
def _root() -> None:
    """TideScout CLI."""


def main() -> None:
    app()
