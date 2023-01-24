# CLI modules

Implementation of CLI modules module

## Camera

Camera module needs to provide support for the following methods:
- Add a camera as a named instance in the system.
    - The camera could be a RTSP or RTMP or HLS stream.
    - Some cameras may have a username/password requriement.
    - Description for camera (Ex: location like OFFICE or Front entrance)

- Preview the camera:
    - Support camera preview through  OpenCV.
    - If this is a headless device - support writing images/videos to disk.
    - Print status on the screen (success/failure, FPS, resolution etc.)

- Remove the camera
    - Remove a named camera instance

- List all the cameras in the system
    - Along with their FPS, description etc.

## Scenario Module

Scenario module needs to provide support for the following methods:
- Listing all scenarios configured for a camera
- Listing all publicly available scenarios
- Listing details about a single scenario
- Add a scenario to a camera
    - Take camera as input
    - Take scenario name as input (validate scenario-name).
    - Take input from user about configuring the scenario. For ex:
        - What time it should run (continuous, specific times of day?)
        - What events should it export (list events --> provide filter like none, summary, recommended, all)
        - In some scenarios - we may have to specify focus area (like exclusion zones etc.). Ask them to input via a CV2 input. Example look [here](https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/)
    - Confirm adding the scenario & then add it.

- List all scenarios
    - List all scenarios for a cameras, or across all cameras.


## Backend

Right now add everything to a local JSON file for storage. Later we can
explore adding to Redis etc. Don't want to add additional dependency during
installation for now.

## Command line or interactive?

I think adding some interactive option would make the program more
usable. For example, this could be one of the end-to-end flows:

```console
$ visionai camera add

Enter name for camera:
OFFICE-01

Enter URI for camera (RTSP or HLS stream - include username/password):
rtsp:admin:password//192.168.0.1/mystream

Description:
This is camera at office facing the door.

-------
Testing camera stream ...... SUCCESS!
Adding camera ..... DONE.
OFFICE-01 Camera successfully added.
```

The same could be implemented by just taking all arguments from command line. I think
if some arguments are not entered - then we can just prompt for it. This would allow for
easy automation scripts later.

```console
$ visionai camera add --camera OFFICE-01 --uri rtsp://admin:password@192.168.0.1/mystream

Description:
This is camera at office facing the door.

-------
Testing camera stream ...... SUCCESS!
Captured 100 frames at 20.2fps.
Adding camera ..... DONE.
OFFICE-01 Camera successfully added.
```

## Can a same URI be added as a new named instance?

Yes. Don't see any advantage for blocking this for now. For example
if OFFICE-01 is added with same RTSP stream, then OFFICE-02 is also
added with same RTSP stream name. Later we can think through this
use-case and if we need to add an error message for it.


## Take the credentials info separately

Yes - it is not ideal to store the camera uri with username/password
in the JSON file without encryption. This would be a feature we should
implement soon. We should prompt user for username/password separately
and store it as a encrypted secret value.


## Scenarios schema

Come up with a simple scenarios schema under config. This is going to be used for configuring each scenarios. Each scenario might have a different requirement - so keep that part flexible.