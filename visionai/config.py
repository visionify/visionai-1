import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # visionai/visionai directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# Config file
CONFIG_FILE = ROOT / 'config' / 'config.json'
