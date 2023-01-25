import typer
from rich import print

web_app = typer.Typer()


# web app
@web_app.command('start')
def web_start():
    '''
    Start web server

    Use this function to start the web-service. Web service
    can be used for more intuitive configuration for the
    cameras and scenarios. Web-app is also the place to view
    event details, camera live-stream etc.
    '''

    print(f'Starting web server')

@web_app.command('stop')
def web_stop(web: str=None):
    '''
    Stop web server

    Use this function to stop already running web-service. There
    can be a single instance of the web-service supported currently.
    So there is no need for any arguments for this function.
    '''

    print(f'Stop web server')

@web_app.command('status')
def web_status():
    '''
    Web service status

    Use this function to get the status of the web-service. (if
    its running or not. This function also prints diagnostic
    information like last few log messages etc.)
    '''
    print(f'Web service status')

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
