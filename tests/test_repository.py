"""Check portability, source integrity and accidental-data-export boundaries."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepositoryTests(unittest.TestCase):
    def test_current_code_correspondence(self):
        self.assertTrue(load_script('verify_code').verify()['passed'])

    def test_cli_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            for command in [['list'], ['train-eac', '--help'], ['validate-three', '--help']]:
                result = subprocess.run([sys.executable, str(ROOT/'scripts/run_workflow.py'), *command],
                                        cwd=folder, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, str(ROOT/'scripts/run_workflow.py'),
                                     'train-eac', '--fold', '8'], cwd=folder, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((Path(folder)/'outputs').exists())

    def test_external_workspace_is_independent_of_code_root(self):
        with tempfile.TemporaryDirectory() as folder:
            env = dict(os.environ, GASTRIC_MTL_WORKSPACE=folder,
                       PYTHONPATH=os.pathsep.join([str(ROOT/'src'), str(ROOT)]))
            result = subprocess.run([sys.executable, '-c',
                'import json; from gastric_mtl import paths; '
                'print(json.dumps({"code": str(paths.ROOT), "workspace": str(paths.WORKSPACE), '
                '"raw": str(paths.RAW_DATA), "pathways": str(paths.PATHWAYS), '
                '"reference": str(paths.REFERENCE_INITIAL), "three": str(paths.THREE_CANCER), '
                '"eac": str(paths.EAC_STAD), "figures": str(paths.FIGURES)}))'],
                env=env, cwd=folder, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            workspace = Path(folder).resolve()
            self.assertEqual(json.loads(result.stdout), {
                'code': str(ROOT), 'workspace': str(workspace),
                'raw': str(workspace/'datasets/raw'), 'pathways': str(workspace/'datasets/pathways'),
                'reference': str(workspace/'results/references/initial'),
                'three': str(workspace/'results/three_cancer'),
                'eac': str(workspace/'results/eac_stad'),
                'figures': str(workspace/'results/figures')})

    def test_command_runner_supports_spawn_workers(self):
        """The dispatcher's __main__ must be pickleable on macOS and Windows."""
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/'spawn_probe.py').write_text(
                'from concurrent.futures import ProcessPoolExecutor\n'
                'from multiprocessing import get_context\n'
                'def worker(value):\n    return value * value\n'
                'if __name__ == "__main__":\n'
                '    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as pool:\n'
                '        assert pool.submit(worker, 7).result(timeout=20) == 49\n')
            (directory/'launch.py').write_text(
                'import sys\n'
                'sys.path.insert(0, '+repr(str(ROOT/'scripts'))+')\n'
                'from run_workflow import execute\n'
                'if __name__ == "__main__": execute("spawn_probe", [])\n')
            result = subprocess.run([sys.executable, str(directory/'launch.py')],
                                    cwd=folder, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_preflight_rejects_incompatible_run_before_loading_data(self):
        from unittest.mock import patch
        from workflows.data import preflight_eac
        with patch.object(preflight_eac, 'initialize', side_effect=RuntimeError('source mismatch')):
            with patch.object(preflight_eac, 'load_submission_cohorts') as load:
                with self.assertRaisesRegex(RuntimeError, 'source mismatch'):
                    preflight_eac.main()
                load.assert_not_called()

    def test_training_cannot_adopt_unversioned_prediction_files(self):
        from unittest.mock import patch
        from workflows.experiments import eac_stad, three_cancer
        for module in [eac_stad, three_cancer]:
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                prediction = directory/'outer_predictions.json'
                prediction.write_text('{"retained": true}')
                with patch.object(module, 'OUT', directory), patch.object(module, 'protocol', return_value={}):
                    with self.assertRaisesRegex(RuntimeError, 'fresh results workspace'):
                        module.initialize()
                self.assertEqual(prediction.read_text(), '{"retained": true}')
                self.assertFalse((directory/'protocol.json').exists())

    def test_export_excludes_research_assets(self):
        files = load_script('export_github').selected_files(ROOT)
        for path in files:
            relative = path.relative_to(ROOT)
            self.assertNotIn(relative.parts[0], {'data', 'datasets', 'outputs', 'results', 'documents', 'archive', 'artifacts'})
            self.assertNotIn(path.suffix.lower(), {'.docx', '.pdf', '.pkl', '.npz', '.gmt', '.zip'})
            self.assertNotIn('__pycache__', relative.parts)

    def test_historical_source_lookup_requires_exact_hash(self):
        from gastric_mtl.paths import verify_recorded_source
        manifest = json.loads((ROOT/'provenance/source_mapping.json').read_text())['original_source_hashes']
        name = 'src/submission_cox.py'
        self.assertEqual(verify_recorded_source(name, manifest[name]), 'reviewed_relocation_not_byte_identity')
        with self.assertRaises(ValueError):
            verify_recorded_source(name, '0'*64)
        with self.assertRaises(ValueError):
            verify_recorded_source('../README.md', '0'*64)

    def test_all_workflow_modules_import_without_loading_cohorts(self):
        import importlib
        os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'gastric_mtl_test_mpl'))
        for _, (module, _) in load_script('run_workflow').WORKFLOWS.items():
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == '__main__':
    unittest.main()
