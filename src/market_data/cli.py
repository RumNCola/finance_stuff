from pathlib import Path

import typer

from .extractor import extract_history
from .query import load_history
from .registry import Registry

app = typer.Typer(help="Historical market data extraction and research storage.")


@app.command()
def download(
    tickers: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    workers: int = 8,
    output: Path = Path("./data"),
) -> None:
    result = extract_history(
        tickers=[t.strip() for t in tickers.split(",")],
        start_date=start,
        end_date=end,
        interval=interval,
        workers=workers,
        output_path=output,
    )
    typer.echo(result.summary())
    if result.errors:
        raise typer.Exit(1)


@app.command()
def status(output: Path = Path("./data")) -> None:
    for row in Registry(output).list():
        typer.echo(" | ".join(str(value) for value in row))


@app.command()
def validate(output: Path = Path("./data")) -> None:
    frame = load_history(output_path=output)
    typer.echo(f"Loaded {frame.height} observations")


@app.command()
def inspect(output: Path = Path("./data")) -> None:
    typer.echo(f"Registry: {Registry(output).path}")


@app.command()
def refresh(tickers: str, output: Path = Path("./data"), interval: str = "1d") -> None:
    """Re-download the requested histories and replace local observations."""
    result = extract_history(
        tickers=[ticker.strip() for ticker in tickers.split(",")],
        interval=interval,
        output_path=output,
        force_refresh=True,
    )
    typer.echo(result.summary())


@app.command()
def benchmark() -> None:
    """Explain how to run the opt-in live benchmark script."""
    typer.echo("Run: python benchmarks/benchmark_download.py")


if __name__ == "__main__":
    app()
