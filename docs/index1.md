# visionai

Documentation for visionai command line interface [visionai.visionify.com](https://visionai.visionify.com).

## Commands

* `visionai scenarios` - Scenario commands
* `visionai cameras` - Camera commands
* `visionai run` - Running the application
* `visionai configure` - Setup up notification and events configuration
* `visionai --help` - Help for the using visionai

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

## Adding additional content

This is some additional content.


## Command overview

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

