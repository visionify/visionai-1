import os
import platform
import sys
from pathlib import Path
import typer
from typing import Optional
import tomli

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # visionai/visionai directory
PKGDIR = FILE.parents[1] # visionai directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from cli import scenario_app, camera_app, web_app, auth_app, device_app, pipeline_app

app = typer.Typer()
app.add_typer(auth_app, name='auth')
app.add_typer(device_app, name='device')
app.add_typer(scenario_app, name='scenario')
app.add_typer(camera_app, name='camera')
app.add_typer(web_app, name='web')
app.add_typer(pipeline_app, name='pipeline')

PYPROJECT_TOML = PKGDIR / 'pyproject.toml'
with open(PYPROJECT_TOML, 'rb') as f:
    pkgdata = tomli.load(f)

__version__ = pkgdata['tool']['poetry']['version']
__verbose__ = False

''' Enable verbose logging '''
def verbose_callback(value: bool):
    if value:
        print('Enabling verbose logging')
        __verbose__ = True

''' Print version & exit '''
def version_callback(value: bool):
    if value:
        print(f'VisionAI Toolkit Version: {__version__}')
        raise typer.Exit()

@app.callback()
def app_callback(
    verbose: Optional[bool]=typer.Option(False, '--verbose', callback=verbose_callback, is_eager=True),
    version: Optional[bool]=typer.Option(False, '--version', callback=version_callback, is_eager=True)
):
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