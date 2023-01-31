import os
import sys
from pathlib import Path
import unittest

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # visionai/visionai directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from util.general import WorkingDirectory, invoke_cmd

class TestDockerCli(unittest.TestCase):
    @WorkingDirectory(ROOT)
    def test_docker_cli(self):
        output = str(invoke_cmd(f'docker ps'))
        print(output)
        print('- - - - - - - - - - - - - - - - - - -')
        for idx, line in enumerate(output.split('\n')):
            print(f'{idx}: {line}')


if __name__ == '__main__':
    dockerObj = TestDockerCli()
    dockerObj.test_docker_cli()


