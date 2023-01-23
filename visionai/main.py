import os
import platform
import sys
from pathlib import Path
import typer
from rich import print
from rich.progress import track

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # visionai/visionai directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from visionai.auth import auth_app
from visionai.device import device_app
from visionai.scenario import scenario_app
from visionai.camera import camera_app
from visionai.web import web_app
from visionai.pipeline import pipeline_app

app = typer.Typer()
app.add_typer(auth_app, name='auth')
app.add_typer(device_app, name='device')
app.add_typer(scenario_app, name='scenario')
app.add_typer(camera_app, name='camera')
app.add_typer(web_app, name='web')
app.add_typer(pipeline_app, name='pipeline')

@app.callback()
def app_callback():
    '''
    VisionAI Toolkit

    VisionAI tookit provides a large number of ready-to-deploy scenarios
    built using latest computer vision frameworks. Supports many of the
    common workplace health and safety use-cases.

    Start by exploring scenarios through visionai scenario list command.
    After that, you can create a pipeline through the pipeline commands.
    Once a pipeline is configured, you can run the pipeline on the
    any number of cameras.

    Running the toolkit does assume a NVIDIA GPU powered machine for
    efficient performance. Please see the system requirements on the
    documentation.

    You can instead opt to install it through Azure Managed VM, with
    preconfigured machines & recommended hardware support. You can
    find information about this on our documentation website.

    Visit https://docs.visionify.ai for more details.
    '''

if __name__ == "__main__":
    app()