import typer
from rich import print
from config import WEB_SERVICE_DOCKER_IMAGE,WEB_SERVICE_DOCKER_HUB_IMAGE
import docker
from docker.errors import *
from rich.console import Console
from rich import print, prompt

err_console = Console(stderr=True)


web_app = typer.Typer()



# web app

@web_app.command('install')
def web_install():
    '''
    Install web app

    Use this function to install the web-service. For now
    sample web app will install later we can have api server
    intigration
    '''
    try:
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - -')
        print("Installing webservice....")
        client = docker.from_env()
        client.images.pull(WEB_SERVICE_DOCKER_HUB_IMAGE,all_tags=False)
    except NotFound as e:
        message = typer.style(e, fg=typer.colors.WHITE, bg=typer.colors.RED)
        typer.echo(message)


@web_app.command('start')
def web_start(
    port: int = typer.Option(8000, help='Port', prompt=True)):
    '''
    Start web server

    Use this function to start the web-service. Web service
    can be used for more intuitive configuration for the
    cameras and scenarios. Web-app is also the place to view
    event details, camera live-stream etc.
    '''
    try:
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - -')
        print(f'Starting web server with port {port}')
        client = docker.from_env()
        client.containers.run(WEB_SERVICE_DOCKER_IMAGE,ports={80:port},detach=True,name='web')
    except NotFound as e:
        print(e)
        message = typer.style(e, fg=typer.colors.WHITE, bg=typer.colors.RED)
        typer.echo(message)
    except ContainerError as e:
        message = typer.style(e, fg=typer.colors.WHITE, bg=typer.colors.RED)
        typer.echo(message)
    except APIError as e:
        message = typer.style(e, fg=typer.colors.WHITE, bg=typer.colors.RED)
        typer.echo(message)


@web_app.command('stop')
def web_stop(web: str=None):
    '''
    Stop web server

    Use this function to stop already running web-service. There
    can be a single instance of the web-service supported currently.
    So there is no need for any arguments for this function.
    '''
    try:
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - -')
        print(f'Stop web server....')
        client = docker.from_env()
        web_service_container = client.containers.get('web')
        web_service_container.stop()
        web_service_container.remove()
    except NotFound :
        message = typer.style(f"No webservice container is running", fg=typer.colors.WHITE, bg=typer.colors.RED)
        typer.echo(message)
    


@web_app.command('status')
def web_status(
    logs: int = typer.Option(5, help='list logs', prompt=True)
    ):
    '''
    Web service status

    Use this function to get the status of the web-service. (if
    its running or not. This function also prints diagnostic
    information like last few log messages etc.)
    '''
    try:
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - -')
        print(f'Web service status....')
        client = docker.from_env()
        web_service_status = client.containers.get('web')
        web_service_status_message = typer.style(web_service_status.status, fg=typer.colors.WHITE, bg=typer.colors.GREEN)
        typer.echo(web_service_status_message)

        logs = web_service_status.logs(tail=logs)
        log_message= logs.decode("utf-8")
        print(log_message)

        web_service_port_message = typer.style(web_service_status.ports, fg=typer.colors.WHITE, bg=typer.colors.GREEN)
        typer.echo(web_service_port_message)

    except NotFound:
        message = typer.style(f"No webservice container is running", fg=typer.colors.WHITE, bg=typer.colors.RED)
        typer.echo(message)


@web_app.callback()
def callback():
    '''
    Web functions

    A web-app is more intuitive way of configuring scenarios,
    cameras and pipelines. These routines allow the user
    to start their own internal development server that
    can be used for managing scenarios, cameras and pipelines.
   '''

if __name__ == '__main__':
    web_app()
