# edgectl

Edge device, cameras & scenarios controller.

## Overview

1. Start the server instance

```bash
make server.install
make server.start
```

1. Run this utility through docker:

```bash
make cli.install
make cli.start

# Run python edgectl.py utility from within container
python3 edgectl.py --get-cameras
```

1. OR Run CLI from host machine

```bash
# Run python edgectl.py utility from host machine
python3 edgectl.py --get-cameras
```

## Commands

The following set of commands are supported.

- TODO: Instead of using IoT Hub connection string - support it via an API key that is generated from our website.
- TODO: Make edgectl into its own repo (public). Create nice documentation on how this can be used in Gitbooks.
- TODO: Connection string to API key translation through Azure Keyvault

After the edgectl configured, we can run the following commands.

```bash
# Configure edgectl
python3 edgectl.py --setup --api-key <api-key>

#  Run edgectl utility
python3 edgectl.py \
    # Device
    --list-devices                          # list devices
    --use-device <device-id>                # set default device to use
    --list-modules                          # list running modules status
    --gpu-stats                             # get device's GPU stats
    --mem-stats                             # get device's memory stats
    --scenarios-health                      # how many more scenarios can run

    # Scenarios
    --list-all-scenarios                    # list all available scenarios
    --list-scenarios                        # list running scenarios for a camera
        --camera <camera-name>
    --start-scenario <scenario-name>        # start a scenario for a camera
        --camera <camera-name>
    --stop-scenario <scenario-name>         # stop a scenario for a camera
        --camera <camera-name>

    # Cameras
    --list-cameras                          # list cameras
    --add-camera <camera-name>              # add a camera
        --camera-uri <camera-uri>
    --remove-camera <camera-name>           # remove camera

    # Livestream
    --start-livestream <camera-name>        # start live-stream for camera
    --stop-livestream <camera-name>         # stop live-stream for camera

    # Events
    --list-events                           # list supported events
        --scenario  <scenario-name>

    --event-details                         # list details for an event
        --scenario <scenario-name>
        --event <event-name>

    --event-log                             # list last few events from camera
        --camera <camera-name>
        --last <duration-seconds>

    # Simulation
    --simulate-event                        # simulate generating an event
        --camera <camera-name>
        --scenario <scenario-name>
        --event <event-name>
        --event-data <event-data>

```
