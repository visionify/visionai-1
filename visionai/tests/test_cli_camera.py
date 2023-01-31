import os
import sys
from pathlib import Path
import unittest

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # visionai/visionai
PKGDIR = FILE.parents[2] # visionai folder
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from config import VISIONAI_EXEC
from util.general import WorkingDirectory, invoke_cmd

class TestInvokeCliCamera(unittest.TestCase):
    # def setUp(self):
    #     # Uninstall package
    #     output = invoke_cmd('pip uninstall -y visionai')

    @WorkingDirectory(PKGDIR)
    def test_invoke_camera(self):
        output = invoke_cmd(f'{VISIONAI_EXEC} camera')
        assert 'Error' in output
        assert 'Missing command' in output

    @WorkingDirectory(PKGDIR)
    def test_invoke_camera_help(self):
        output = invoke_cmd(f'{VISIONAI_EXEC} camera --help')
        assert 'Usage' in output
        assert 'Commands' in output
        assert 'add' in output
        assert 'list' in output
        assert 'remove' in output

    @WorkingDirectory(PKGDIR)
    def test_invoke_camera_add_help(self):
        output = invoke_cmd(f'{VISIONAI_EXEC} camera add --help')
        assert 'Usage' in output
        assert 'camera' in output
        assert 'add' in output
        assert '--name' in output
        assert '--uri' in output
        assert '--description' in output
        assert '--help' in output

    @WorkingDirectory(PKGDIR)
    def test_invoke_camera_add_remove(self):
        # cleanup (prior test failures)
        output = invoke_cmd(f'{VISIONAI_EXEC} camera remove --name TEST-999')

        # add camera
        output = invoke_cmd(f'{VISIONAI_EXEC} camera add --name TEST-999 --uri youtube.com --description "Test camera"')
        assert 'Success' in output

        # list camera
        output = invoke_cmd(f'{VISIONAI_EXEC} camera list')
        assert "'name': 'TEST-999'" in output
        assert "'uri': 'youtube.com'" in output

        # remove camera
        output = invoke_cmd(f'{VISIONAI_EXEC} camera remove --name TEST-999')
        assert 'Success' in output

        # list camera
        output = invoke_cmd(f'{VISIONAI_EXEC} camera list')
        assert "'name': 'TEST-999'" not in output


if __name__ == '__main__':
    unittest.main()
