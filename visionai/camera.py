import typer
import time
from rich.progress import track

camera_app = typer.Typer()


# Camera app
@camera_app.command('list')
def camera_list():
    cameras = ['OFFICE-01', 'OFFICE-02']
    print(f'Cameras : {cameras}')

@camera_app.command('add')
def camera_add(camera: str):
    print(f'Adding camera : {camera}')

@camera_app.command('remove')
def camera_remove(camera: str):
    print(f'Removing camera : {camera}')

@camera_app.command('install')
def camera_install(
    name:str = typer.Option(..., help='camera name to install'),
    uri: str = typer.Option(..., help='camera uri (rtsp/hls/http)')
    ):
    total = 0
    for value in track(range(100), description="Adding camera..."):
        # Fake processing time
        time.sleep(0.01)
        total += 1
    print(f"Processed {total} things.")

@camera_app.callback()
def callback():
    '''
    Manage adding/removing cameras

    Users can add a camera through its name and uri

    A new camera will be created in the system.
    '''

if __name__ == '__main__':
    camera_app()
