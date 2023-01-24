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
    def test_invoke_main(self):
        output = invoke_cmd(f'python main.py')
        assert 'Error' in output
        assert 'Missing command' in output

    @WorkingDirectory(ROOT)
    def test_invoke_main_help(self):
        output = invoke_cmd('python main.py --help')
        assert 'VisionAI Toolkit' in output
        assert 'docs.visionify.ai' in output

if __name__ == '__main__':
    unittest.main()
