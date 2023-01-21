import typer

device_app = typer.Typer()

# Device app
@device_app.command('list')
def device_list():
    '''
    List available devices

    Get a list of all available [processing] devices
    '''
    devices = ['oos-training', 'edge-dev1', 'edge-dev2']
    print(f'Devices : {devices}')

@device_app.command('select')
def device_select(device: str):
    '''
    Select a device

    Not sure why is this needed at this time.
    '''
    print(f'Selecting device : {device}')

@device_app.command('modules')
def device_modules():
    '''
    List running modules on the device

    Again this does not make much sense at this time. Let's revisit.
    '''
    modules = ['edgeCtl', 'edgeEvents']
    print(f'Listing modules : {modules}')

@device_app.command('stats')
def device_gpu_mem_stats():
    '''
    Machine health (GPU/Mem stats)

    Show machine health (GPU/memory stats). This can be used to
    determine if more scenarios can be run on the machine or not.
    '''
    print('Getting gpu/mem stats')

if __name__ == '__main__':
    device_app()
