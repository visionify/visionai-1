import typer
from rich import print

scenario_app = typer.Typer()



# Scenario app
@scenario_app.command('list')
def scenario_list():
    scenarios = ['smoke-and-fire', 'ppe-detection', 'edge-dev2']
    print(f'Scenarios : {scenarios}')

@scenario_app.command('add')
def scenario_add(scenario: str):
    print(f'Adding scenario : {scenario}')

@scenario_app.command('remove')
def scenario_remove(scenario: str):
    print(f'Removing scenario : {scenario}')


@scenario_app.callback()
def callback():
    '''
    Manage scenarios

    Browse scenarios. Get details of a scenario. Get list
    of scenarios available for a tag or category. Get model
    details for a scenario.
   '''


if __name__ == '__main__':
    scenario_app()
