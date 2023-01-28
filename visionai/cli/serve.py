import os
import sys
import typer
import time
import json
from uuid import uuid4
from rich import print, prompt
from rich.progress import track
from pathlib import Path
import requests

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # visionai/visionai directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

MODELS_REPO = ROOT / 'models-repo'

from config import CONFIG_FILE, SCENARIOS_SCHEMA, SCENARIOS_URL, TRITON_SERVER_DOCKER_IMAGE
from util.general import invoke_cmd
from util.download_models import safe_download_to_folder

# Model serve app
serve_app = typer.Typer()

@serve_app.command('start')
def serve_start():
    '''
    Start serving all models.

    All models present in the models-repo will be served. We use
    triton inference server to serve them. The triton server will be
    at http://localhost:8000, grpc://localhost:8001.

    Please make sure these two ports are not used by anyone else.
    '''
    print('Start model serve...')
    print(f'Models repo: {MODELS_REPO}')
    models_to_serve = []
    files_folders = os.listdir(MODELS_REPO)
    for file_folder in files_folders:
        if os.path.isdir(file_folder):
            models_to_serve.append(file_folder)

    print(f'Found {len(models_to_serve)} models to serve.')
    if len(models_to_serve) > 0:
        for model_to_serve in models_to_serve:
            print(f'.. {model_to_serve}')
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - -')

    print(f'Pulling docker image: {TRITON_SERVER_DOCKER_IMAGE}')
    invoke_cmd(f'docker pull {TRITON_SERVER_DOCKER_IMAGE}')
    print(f'Docker image pull complete.')
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - -')

    print('Starting docker image')
    docker_cmd = f'docker run -d --gpus=1 --rm --net=host -v {MODELS_REPO}:/models {TRITON_SERVER_DOCKER_IMAGE} tritonserver --model-repository=/models'
    print(f'Command used: \n{docker_cmd}')
    invoke_cmd(docker_cmd)
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('Started serve container')
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - -')


    # TODO: would be good to check docker status somehow & show it here.



@serve_app.command('stop')
def serve_stop():
    '''
    Stop serving all models.

    This method will stop serving all models. Any inference running
    will all be stopped as well.
    '''
    print('Stop model serve...')



@serve_app.command('status')
def serve_status():
    '''
    Show the status of server

    Shows how many models are being served, metrics for the models etc.
    '''
    print('Model serve status: ')

@serve_app.callback()
def callback():
    '''
    Serve models through Triton

    We use Triton inference server to make the best use of GPU/CPU
    resources available on the machine in order to serve our models. Any
    models that are available in models-repo folder would be served. All
    models are served, if there is any failure in one of the models - then
    no other models are also served.
    '''

if __name__ == '__main__':
    serve_app()

