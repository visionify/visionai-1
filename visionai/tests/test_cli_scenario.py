import os
import sys
from pathlib import Path
import unittest

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # visionai/visionai directory
PKGDIR = FILE.parents[2] # visionai dir
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from util.general import WorkingDirectory, invoke_cmd

class TestInvokeCliScenario(unittest.TestCase):
    def setUp(self):
            # Uninstall package
        output = invoke_cmd('pip uninstall -y visionai')

    @WorkingDirectory(PKGDIR)
    def test_invoke_scenario(self):
        output = invoke_cmd(f'python -m visionai scenario')
        assert 'Error' in output
        assert 'Missing command' in output

    @WorkingDirectory(PKGDIR)
    def test_invoke_scenario_help(self):
        output = invoke_cmd('python -m visionai scenario --help')
        assert 'Usage' in output
        assert 'Commands' in output
        assert 'add' in output
        assert 'list' in output
        assert 'remove' in output

    @WorkingDirectory(PKGDIR)
    def test_invoke_scenario_add_help(self):
        output = invoke_cmd('python -m visionai scenario add --help')
        assert 'Usage' in output
        assert 'scenario' in output
        assert 'add' in output
        assert '--camera' in output
        assert '--scenario' in output
        assert '--help' in output

    @WorkingDirectory(PKGDIR)
    def test_invoke_scenario_remove_help(self):
        output = invoke_cmd('python -m visionai scenario remove --help')
        assert 'Usage' in output
        assert 'scenario' in output
        assert 'remove' in output
        assert '--scenario' in output
        assert '--camera' in output
        assert '--help' in output

    @WorkingDirectory(PKGDIR)
    def test_invoke_scenario_list_help(self):
        output = invoke_cmd('python -m visionai scenario list --help')
        assert 'Usage' in output
        assert 'scenario' in output
        assert '--camera' in output
        assert '--help' in output

    @WorkingDirectory(PKGDIR)
    def test_invoke_scenario_add_remove(self):
        # cleanup (prior test failures)
        output = invoke_cmd('python -m visionai scenario remove --name TEST-999')

        # add camera
        output = invoke_cmd('python -m visionai camera add --name TEST-999 --uri youtube.com --description "Test camera"')
        assert 'Success' in output

        # add scenario
        output = invoke_cmd(f'python -m visionai scenario add --camera TEST-999 --scenario smoke-and-fire-detection')
        assert 'Scenario' in output
        assert 'smoke-and-fire-detection' in output
        assert 'added for camera' in output

        # list scenario
        output = invoke_cmd('python -m visionai scenario list --camera TEST-999')
        assert "Listing configured scenarios for" in output
        assert 'TEST-999' in output
        assert "'name': 'smoke-and-fire-detection'" in output

        # remove scenario
        output = invoke_cmd('python -m visionai scenario remove --camera TEST-999 --scenario smoke-and-fire-detection')
        assert 'Success' in output

        # list scenario
        output = invoke_cmd('python -m visionai scenario list  --camera TEST-999')
        assert "Listing configured scenarios for" in output
        assert 'TEST-999' in output
        assert "'name': 'smoke-and-fire-detection'" not in output

        # remove camera
        output = invoke_cmd('python -m visionai scenario remove --name TEST-999')

    @WorkingDirectory(PKGDIR)
    def test_list_all_scenarios(self):
        # list scenario
        output = invoke_cmd('python -m visionai scenario list-all')
        assert 'Detect early signs of' in output
        assert 'ppe-detection' in output
        assert 'model_url' in output
        assert 'categories' in output
        assert 'tags' in output
        assert 'accuracy' in output
        assert 'recall' in output
        assert 'f1' in output
        assert 'datasetSize' in output

    @WorkingDirectory(PKGDIR)
    def test_list_no_camera_list_all_scenarios(self):
        # list scenario
        output = invoke_cmd('python -m visionai scenario list')
        assert 'Detect early signs of' in output
        assert 'ppe-detection' in output
        assert 'model_url' in output
        assert 'categories' in output
        assert 'tags' in output
        assert 'accuracy' in output
        assert 'recall' in output
        assert 'f1' in output
        assert 'datasetSize' in output

    @WorkingDirectory(PKGDIR)
    def test_download_specific_scenario(self):
        # download specific scenario
        output = invoke_cmd('python -m visionai scenario download --scenario smoke-and-fire-detection')
        assert 'download_models' in output
        assert 'Model: smoke-and-fire-detection' in output

    @WorkingDirectory(PKGDIR)
    def test_download_all_scenarios(self):
        # download all scenarios
        output = invoke_cmd('python -m visionai scenario download')
        assert 'Downloading scenarios : all' in output

if __name__ == '__main__':
    unittest.main()
