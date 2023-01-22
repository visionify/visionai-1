import os
import platform
import sys
from pathlib import Path
import typer
from rich import print
from rich.progress import track

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # visionai root directory
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

    Start by exploring scenarios
    ```
    visionai scenario list
    visionai scenario <scenario-name> details
    ```

    VisionAI toolkit also supports running multiple scenarios or a sequence
    of scenarios. Run scenarios on multiple IP cameras in through
    `visionai run` command.

    Following are the sub-commands supported. You can get more
    details about each of the scenarios using `visionai <command> --help`

    ```
    visionai scenario       -- Listing scenarios, categories, tags & detailsListing details of a scenario
    visionai camera         -- Add/remove named camera streams.
    visionai pipelin3       -- Configure and run pipelines
    visionai device         -- Device stats (GPU/Memory)
    visionai web            -- Start/stop web-service to manage this
    visionai auth           -- Login/logout through API token for private models
    ```
    '''


if __name__ == "__main__":
    app()