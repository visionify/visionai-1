
import requests
import json
from pathlib import Path

FILE = Path(__file__).resolve()
CONFIGDIR = FILE.parents[0]  # config directory

def main():
    SCHEMA_URL = 'https://raw.githubusercontent.com/visionify/visionai-scenarios/main/scenarios.json'
    SCHEMA_FILE = CONFIGDIR / 'scenario-schema.json'
    res = requests.get(SCHEMA_URL)
    scenarios = res.json()
    with open(SCHEMA_FILE, 'w') as f:
        json.dump(scenarios, f, indent=4)

    print(f'Scenarios schema {SCHEMA_URL} downloaded to {SCHEMA_FILE}')

if __name__ == '__main__':
    main()
