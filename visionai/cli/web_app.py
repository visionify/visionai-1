import typer
from rich import print
from config import WEB_SERVICE_DOCKER_HUB_IMAGE, WEB_SERVICE_PORT
import docker
from docker.errors import *
from rich.console import Console
from rich import print, prompt

from util.docker_utils import docker_image_pull_with_progress

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
        docker_image_pull_with_progress(client, WEB_SERVICE_DOCKER_HUB_IMAGE)
        print("Webservice installed successfully")
    except NotFound as e:
        message = typer.style(e, fg=typer.colors.WHITE, bg=typer.colors.RED)
        typer.echo(message)


@web_app.command('start')
def web_start(
    port: int = typer.Option(WEB_SERVICE_PORT, help='Port')):
    '''
    Start web server

    Use this function to start the web-service. Web service
    can be used for more intuitive configuration for the
    cameras and scenarios. Web-app is also the place to view
    event details, camera live-stream etc.
    '''
    try:
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - -')

        # check if already running
        client = docker.from_env()
        containers = client.containers.list()
        for container in containers:
            if container.name == 'visionai-web':
                print(f'Web server already running at: http://localhost:{port}')
                return

        # start web server
        print(f'Starting web server at port {port}')
        client = docker.from_env()
        client.containers.run(
            WEB_SERVICE_DOCKER_HUB_IMAGE,
            ports={80:port},
            detach=True,
            name='visionai-web')
        print(f'Web server available at: http://localhost:{port}')

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
        web_service_container = client.containers.get('visionai-web')
        web_service_container.stop()
        web_service_container.remove()
    except NotFound :
        message = typer.style(f"Web-server not running", fg=typer.colors.WHITE, bg=typer.colors.RED)
        typer.echo(message)



@web_app.command('status')
def web_status(
    tail: int = typer.Option(20, help='tail number of lines')
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
        web_service_status = client.containers.get('visionai-web')
        web_service_status_message = typer.style(web_service_status.status, fg=typer.colors.WHITE, bg=typer.colors.GREEN)
        typer.echo(web_service_status_message)

        logs = web_service_status.logs(tail=tail)
        log_message= logs.decode("utf-8")
        print(log_message)

        web_service_port_message = typer.style(web_service_status.ports, fg=typer.colors.WHITE, bg=typer.colors.GREEN)
        typer.echo(web_service_port_message)

    except NotFound:
        message = typer.style(f"Web-server not running", fg=typer.colors.WHITE, bg=typer.colors.RED)
        typer.echo(message)


@web_app.callback()
def callback():
    '''
    Start/stop web-app

    Start and stop the VisionAI web-app which can be a more
    intuitive way of managing cameras, pipelines and scenarios.
    Web-app also provides a live-stream view of the cameras.
    '''
    pass

if __name__ == '__main__':
    web_app()

    # web_install()