import os
import platform
import sys
from pathlib import Path
import typer
from rich import print
from rich.progress import track
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # edgectl root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from visionai.auth import auth_app
from visionai.device import device_app
from visionai.scenario import scenario_app
from visionai.camera import camera_app

app = typer.Typer()
app.add_typer(auth_app, name='auth')
app.add_typer(device_app, name='device')
app.add_typer(scenario_app, name='scenario')
app.add_typer(camera_app, name='camera')

@app.callback()
def app_callback():
    '''
    edgectl utility

    This utility can be used to manage cameras and scenarios.

    You can add and remove cameras. or you can add and remove scenarios
    with this application. This application can also be used to restart
    the edge device or the containers running there. Try out and let us
    know if you run into any issues.
    '''


if __name__ == "__main__":
    app()