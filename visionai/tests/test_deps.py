
import os
import sys
from pathlib import Path
import unittest

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # visionai/visionai directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from util.general import WorkingDirectory, invoke_cmd

class TestDeps(unittest.TestCase):
    @WorkingDirectory(ROOT)
    def test_import_torch(self):
        output = invoke_cmd(f'python -c "import torch; print(torch.__version__)"')
        assert output.startswith('1.1') # torch 1.12 above

    @WorkingDirectory(ROOT)
    def test_import_cv2(self):
        output = invoke_cmd(f'python -c "import cv2; print(cv2.__version__)"')
        assert output.startswith('4.') # cv2 4.6 or above

if __name__ == '__main__':
    unittest.main()
