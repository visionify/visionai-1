import os
import sys
from pathlib import Path
import unittest

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # visionai/visionai directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from util.general import WorkingDirectory, invoke_cmd

class TestInvoke(unittest.TestCase):
    @WorkingDirectory(ROOT)
    def test_invoke_camera(self):
        output = invoke_cmd(f'python main.py camera')
        assert 'Error' in output
        assert 'Missing command' in output

    @WorkingDirectory(ROOT)
    def test_invoke_camera_help(self):
        output = invoke_cmd('python main.py camera --help')
        assert 'Usage' in output
        assert 'Commands' in output
        assert 'add' in output
        assert 'list' in output
        assert 'remove' in output

    @WorkingDirectory(ROOT)
    def test_invoke_camera_add_help(self):
        output = invoke_cmd('python main.py camera --help')
        assert 'Usage' in output
        assert 'camera' in output
        assert 'add' in output
        assert 'list' in output
        assert 'remove' in output
        assert '--help' in output

    @WorkingDirectory(ROOT)
    def test_invoke_camera_add(self):
        output = invoke_cmd('python main.py camera add --name TEST-999 --uri youtube.com --description "Test camera"')
        assert 'Success' in output

    @WorkingDirectory(ROOT)
    def test_invoke_camera_remove(self):
        output = invoke_cmd('python main.py camera remove --name TEST-999')
        assert 'Success' in output


if __name__ == '__main__':
    unittest.main()
