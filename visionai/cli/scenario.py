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

from config import CONFIG_FILE, SCENARIOS_SCHEMA, SCENARIOS_URL
from util.download_models import safe_download_to_folder

scenario_app = typer.Typer()

# scenario app
@scenario_app.command('list')
def scenario_all():
    '''
    List all scenarios available

    List all scenarios available in the system. This includes scenarios
    that may or maynot be applied to any specific camera.
    '''
    try:
        res = requests.get(SCENARIOS_URL)
        # with open(SCENARIOS_SCHEMA, 'w') as f:
        #     json.dump(res.json(), f, indent=4)

        scenarios = res.json()['scenarios']
    except Exception as ex:
        scenarios = str(ex)

    print(scenarios)

@scenario_app.command('download')
def scenario_download(
    scenario: str=typer.Option('all', help='scenario name'),
    world: bool=typer.Option(False, help='Download all public scenarios available')
    ):
    '''
    Download models for scenarios

    all - all scenarios configured for cameras in this system.
    world - all available public scenarios (this might take a lot of space.)
    individual - specify scenario name you want to download.

    Download models for a given scenario, or download models for
    all scenarios that have been configured.
    '''

    print(f'Downloading scenarios : {scenario}')

    # Get list of available scenarios.
    res = requests.get(SCENARIOS_URL)
    all_scenarios = res.json()['scenarios']


    if world is True:
        yN = prompt.Confirm.ask('Are you sure you want to download all scenarios? This may take a lot of space!')
        if yN is True:
            num_scenarios = len(all_scenarios)
            for idx, scen in enumerate(all_scenarios):
                print(f'Downloading {idx+1}/{num_scenarios}...')
                scen_url = scen['models']['latest']['model_url']
                safe_download_to_folder(scen_url, MODELS_REPO)
        else:
            raise typer.Exit()

    if scenario == 'all':
        model_names = set()

        if not os.path.exists(CONFIG_FILE):
            print('No scenarios configured. Please use visionai scenario add first')
            return

        with open(CONFIG_FILE) as f:
            config_data = json.load(f)

        for cam in config_data['cameras']:
            for scen in cam['scenarios']:
                model_names.add(scen['name'])

            for preproc in cam['preprocess']:
                model_names.add(preproc['name'])

        if len(model_names) == 0:
            # No models to download
            print(f'Error: No scenarios configured. Use `visionai scenario add`')

        else:
            # Download requred models
            for scen in all_scenarios:
                scen_name = scen['name']
                scen_url = scen['models']['latest']['model_url']
                if scen_url is None:
                    continue

                if scen_name in model_names:
                    print(f'Model: {scen_name}: {scen_url}')
                    safe_download_to_folder(scen_url, MODELS_REPO)

    else:
        # If a single scenario name is specified.
        for scen in all_scenarios:
            scen_name = scen['name']
            scen_url = scen['models']['latest']['model_url']
            if scen_url is None:
                continue

            if scen_name == scenario:
                print(f'Model: {scen_name}: {scen_url}')
                safe_download_to_folder(scen_url, MODELS_REPO)
                break

    # Done downloading models.
    print('Done.')

@scenario_app.command('preview')
def scenario_preview(
    name:str = typer.Option(..., help='scenario name to preview')
    ):
    '''
    Preview the scenario feed

    View the scenario feed, review FPS etc available for scenario.
    '''
    print(f"TODO: Implement scenario preview functionality.")

@scenario_app.callback()
def callback():
    '''
    Manage scenarios

    An organization can have multiple scenarios that are installed
    at different places. They may be from different vendors and/or
    maybe using different security surveillance software. Most
    scenarios however do support RTSP, RTMP or HLS streams as an
    output. Please refer to your scenario vendor documentation to
    find this out. This module will help you onboard those scenarios
    on visionai systems by using a simple named instance for each
    scenario.
    '''

if __name__ == '__main__':
    # scenario_app()

    # scenario_remove('smoke-and-fire-detection', 'TEST-999')
    # scenario_add('smoke-and-fire', 'TEST-999')
    scenario_download(world=True)
