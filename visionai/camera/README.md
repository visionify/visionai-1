# Camera module

Implementation of camera module

## Exported methods

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




