import os
import platform
import sys
from pathlib import Path
import typer

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # edgectl root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from util.general import print_args, check_requirements, LOGGER

app = typer.Typer()

@app.command()
def list_devices():

def run(
    list_devices=False,
    use_device='oos-training',
    list_modules=False,
    gpu_stats=False,
    mem_stats=False,
    scenario_health=False,
    list_all_scenarios=False,
    list_scenarios=False,
    start_scenario=False,
    stop_scenario=False,
    camera=None,
    list_all_cameras=False,
    add_camera=None,
    camera_uri=None,
    remove_camera=None,
    start_livestream=None,
    stop_livestream=None,
    simulate_events=False,
    scenario=None,
    event=None,
    event_data=None
):
    print(f'list_devices {list_devices}')
    print(f'use_device {use_device}')
    print(f'list_modules {list_modules}')
    print(f'gpu_stats {gpu_stats}')
    print(f'mem_stats {mem_stats}')
    print(f'scenario_health {scenario_health}')
    print(f'list_all_scenarios {list_all_scenarios}')
    print(f'list_scenarios {list_scenarios}')
    print(f'start_scenario {start_scenario}')
    print(f'stop_scenario {stop_scenario}')
    print(f'camera {camera}')
    print(f'list_all_cameras {list_all_cameras}')
    print(f'add_camera {add_camera}')
    print(f'camera_uri {camera_uri}')
    print(f'remove_camera {remove_camera}')
    print(f'start_livestream {start_livestream}')
    print(f'stop_livestream {stop_livestream}')
    print(f'simulate_events {simulate_events}')
    print(f'scenario {scenario}')
    print(f'event {event}')
    print(f'event_data {event_data}')

    if list_devices:
        list_all_devices()
    elif use_device is not None:
        fix_default_device(use_device)
    elif list_modules:
        required()
        list_all_modules()
    elif

def parse_opt():
    parser = argparse.ArgumentParser()

    # Device commands
    devices = parser.add_argument_group('Device')
    devices.add_argument('--list-devices', action='store_true', help='show available devices')
    devices.add_argument('--use-device', type=str, default=None, help='fix default device')
    devices.add_argument('--list-modules', action='store_true', help='show available modules on device')
    devices.add_argument('--gpu-stats', action='store_true', help='show gpu stats for the edgedevice')
    devices.add_argument('--mem-stats', action='store_true', help='show mem stats for the device')
    devices.add_argument('--scenario-health', action='store_true', help='show scenario health for the device')

    # List all scenarios
    list_scenarios = parser.add_argument_group('List scenarios')
    list_scenarios.add_argument('--list-all-scenarios', action='store_true', help='list all available scenarios')

    # Scenario commands
    scenarios = parser.add_argument_group('Start/stop scenarios for a camera')
    scenarios.add_argument('--list-scenarios', action='store_true', help='list scenarios for camera [--camera CAM]')
    scenarios.add_argument('--start-scenario', action='store_true', help='start scenarios for camera [--camera CAM]')
    scenarios.add_argument('--stop-scenario', action='store_true', help='stop scenarios for camera [--camera CAM]')
    scenarios.add_argument('--camera', type=str, help='specify camera name')

    # List all cameras
    list_cameras = parser.add_argument_group('List all cameras')
    list_cameras.add_argument('--list-all-cameras', action='store_true', help='list all available cameras')

    # Add a camera
    add_camera = parser.add_argument_group('Add camera')
    add_camera.add_argument('--add-camera', type=str, help='add camera  [--camera CAM  --camera-uri URI]')
    add_camera.add_argument('--camera-uri', type=str, help='Specify camera URI (rtsp, rtmp, hls, https)')

    # Remove camera
    remove_camera = parser.add_argument_group('Remove camera')
    remove_camera.add_argument('--remove-camera', type=str, help='remove camera  [--camera CAM]')

    # Start/stop livestream
    start_livestream = parser.add_argument_group('Start/stop livestream')
    start_livestream.add_argument('--start-livestream', type=str, help='start livestream [--camera CAM]')
    start_livestream.add_argument('--stop-livestream', type=str, help='stop livestream [--camera CAM]')

    # Simulate event
    simulate_events = parser.add_argument_group('Simulate events')
    simulate_events.add_argument('--simulate-events', action='store_true', help='simulate an event')
    simulate_events.add_argument('--scenario', type=str, help='specify scenario')
    simulate_events.add_argument('--event', type=str, help='event name')
    simulate_events.add_argument('--event-data', type=str, help='event data json in "double quotes"')

    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def main():
    opt = parse_opt()
    run(**vars(opt))

    # check_requirements()
    # run(**vars(opt))

if __name__ == "__main__":
    main()