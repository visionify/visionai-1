
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
        assert "1.1" in output # torch 1.12 or 1.13

    @WorkingDirectory(ROOT)
    def test_import_cv2(self):
        output = invoke_cmd(f'python -c "import cv2; print(cv2.__version__)"')
        assert "4.6" in output # cv2 4.6 or above

if __name__ == '__main__':
    unittest.main()
