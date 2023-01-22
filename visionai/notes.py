import typer
from rich import print
from typing import Optional
from pathlib import Path

app = typer.Typer()

# For taking input as a file or a directory
def main(config: Optional[Path] = typer.Option(None)):
    if config is None:
        print("No config file")
        raise typer.Abort()
    if config.is_file():
        text = config.read_text()
        print(f"Config file contents: {text}")
    elif config.is_dir():
        print("Config is a directory, will use all its config files")
    elif not config.exists():
        print("The config doesn't exist")


# To enable auto-complete
def complete_name():
    return ["Camila", "Carlos", "Sebastian"]

@app.command()
def main(
    name: str = typer.Option(
        "World", help="The name to say hi to.", autocompletion=complete_name
    )
):
    print(f"Hello {name}")


# Progress bar
import time
from rich.progress import track
def main():
    total = 0
    for value in track(range(100), description="Processing..."):
        # Fake processing time
        time.sleep(0.01)
        total += 1
    print(f"Processed {total} things.")


if __name__ == "__main__":
    app()