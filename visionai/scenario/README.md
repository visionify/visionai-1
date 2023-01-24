# Scenario module

Implementation of scenario module

## Exported methods

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

Keep adding all of these details as interactive as possible. But allow
automation scripts by adding command line options.


## Scenarios schema

Come up with a simple scenarios schema under config. This is going to be used for configuring each scenarios. Each scenario might have a different requirement - so keep that part flexible.