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
from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # visionai/visionai directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

MODELS_REPO = ROOT / 'models-repo'

from config import CONFIG_FILE, SCENARIOS_SCHEMA, SCENARIOS_URL
from util.download_models import safe_download_to_folder
from models.triton_client import TritonClient
from scenarios import load_scenario

# scenario app
scenario_app = typer.Typer()

# print primitives
def _scenario_pretty(scenario):
    if scenario is None:
        return Panel('[red]NONE[/red]')

    id = scenario.get('id')
    name = scenario.get('name')

    try:
        version = scenario.get('version')
        overview = scenario.get('overview')[:150] + '...'
        model = scenario.get('models')['latest']['name'] + 'v.' + scenario.get('models')['latest']['version']
        accuracy = scenario.get('models')['latest']['accuracy']
        recall = scenario.get('models')['latest']['recall']
        f1 = scenario.get('models')['latest']['f1']
        datasetSize = scenario.get('models')['latest']['datasetSize']
        docs = scenario.get('docs')
        events = '[b]Events: [/b]' + ' | '.join(scenario.get('events'))
        categories = '[b]Categories: [/b]' + ' | '.join(scenario.get('categories'))
        tags = '[b]Tags: [/b]' + ' | '.join(scenario.get('tags'))
        metrics_txt = f'[magenta]Acc {accuracy}% | Rec: {recall}% | F1: {f1}%[/magenta]\n' + \
            f'[magenta]Size: {datasetSize} images[/magenta]'
        scenario_txt = f'[blue]{model}[/blue]\n' + \
            f'[grey7]{overview}[/grey7]\n' + \
            metrics_txt + '\n' +\
            f'[grey]{docs}[/grey]\n' +\
            f'[cyan]{events}[/cyan]\n' + \
            f'[cyan]{categories}[/cyan]\n' +\
            f'[cyan]{tags}[/cyan]'

        return Panel(scenario_txt, title=f'[yellow]{name}[/yellow]', width=60, expand=False)
    except Exception as ex:
        return Panel(f'[red]ERR {name}[/red]')

def print_scenarios(scenarios):
    console = Console()
    scen_pretty = [_scenario_pretty(scen) for scen in scenarios]
    console.print(Panel(Columns(scen_pretty), title='Scenarios'))


@scenario_app.command('list')
def scenario_list():
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

    print_scenarios(scenarios)

@scenario_app.command('download')
def scenario_download(
    scenario: str=typer.Option('all', help='scenario name'),
    world: bool=typer.Option(False, help='Download all public scenarios available')
    ):
    '''
    Download models for scenarios

    --scenario [NAME] : specify scenario to download.
    --scenario [all]  : download all scenarios configured for the system.
    --scenarios[world]: download all available scenarios.

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


@scenario_app.command('test')
def scenario_test(
    name:str = typer.Option(..., help='scenario name to test'),
    camera: str = typer.Option('0', help='Camera name (default is webcam)')
    ):
    '''
    Run the scenario locally to test it out.

    - Download the model if not available.
    - Start the model server with only this model.
    -
    '''

    res = requests.get(SCENARIOS_URL)
    all_scenarios = res.json()['scenarios']
    scenario_to_test = None
    for scen in all_scenarios:
        if scen['name'] == name:
            scenario_to_test = scen
            break

    if scenario_to_test is None:
        print(f'ERROR: Scenario {name} not found.')
        raise typer.Abort()

    print(f'Downloading models for {name}')
    scenario_download(scenario=name)

    print(f'Starting model server..')
    tc = TritonClient()
    tc.start_model_server()
    tc.print_models_served()

    print(f'Loading inference engine for {name}')
    inference_engine = load_scenario(scenario_name=name, camera_name=camera)

    # for test command - block the main thread.
    print(f'Running on default web-cam')
    inference_engine.start()



    #TODO: Provide support for running on any camera




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
    scenario_app()
    # scenario_list()

    # scenario_remove('smoke-and-fire-detection', 'TEST-999')
    # scenario_add('smoke-and-fire', 'TEST-999')
    # scenario_download(world=True)
