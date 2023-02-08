import os
import sys
from pathlib import Path
import unittest

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # visionai/visionai directory
PKGDIR = FILE.parents[2] # visionai dir
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from config import VISIONAI_EXEC
from util.general import WorkingDirectory, invoke_cmd

class TestInvokeCliWeb(unittest.TestCase):

    @WorkingDirectory(PKGDIR)
    def test_invoke_web_help_command(self):
        output = invoke_cmd(f'{VISIONAI_EXEC} web --help')
        assert 'install' in output
        assert 'start' in output
        assert 'status' in output
        assert 'stop' in output

    @WorkingDirectory(PKGDIR)
    def test_invoke_web_install_command(self):
        output = invoke_cmd(f'{VISIONAI_EXEC} web install')
        assert 'Pulling image: ' in output
        assert 'Webservice installed successfully' in output

    @WorkingDirectory(PKGDIR)
    def test_invoke_web_start_stop_status_command(self):
        output = invoke_cmd(f'{VISIONAI_EXEC} web install')
        assert 'Pulling image: ' in output
        assert 'Webservice installed successfully' in output

        output = invoke_cmd(f'{VISIONAI_EXEC} web start')
        assert 'Starting web server at port' in output
        assert 'Web server available at' in output

        output = invoke_cmd(f'{VISIONAI_EXEC} web status')
        assert 'Web service status' in output
        assert 'running' in output

        output = invoke_cmd(f'{VISIONAI_EXEC} web stop')
        assert 'Stop web server' in output

        output = invoke_cmd(f'{VISIONAI_EXEC} web status')
        assert 'Web service status' in output
        assert 'Web-server not running' in output


if __name__ == '__main__':
    unittest.main()
