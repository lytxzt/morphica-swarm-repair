"""Release regressions; run with python tests/test_release.py."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
import run_control_suites as control
import run_ablation_cohort_b as cohort


class ReleaseChecks(unittest.TestCase):
    def test_cpu_methods_without_learning_packages(self):
        code = '''
import importlib.abc, sys
class NoLearningPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_geometric', 'einops'}:
            raise ModuleNotFoundError('optional package absent', name=fullname)
sys.meta_path.insert(0, NoLearningPackages())
sys.argv = ['cpu-check', '--num-agents', '200']
import numpy as np
from Swarm import Swarm
from Configurations import config_initial_swarm_positions
from tools.damage_mask import damage_mask
for mode in [1, 2, 3, 14]:
    swarm = Swarm(algorithm_mode=mode)
    swarm.destroy_happens(damage_mask(200, 100, 4), config_initial_swarm_positions)
    actions, _ = swarm.take_actions()
    assert actions.shape == (200, 3) and np.isfinite(actions).all()
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_archived_cohort_includes_no_p_failure(self):
        tasks = [('ablation_cohort_b', 1000, 500, case, variant)
                 for case in cohort.CASES for variant in cohort.VARIANTS]
        rows = control.verify_archive(cohort.ARCHIVED, tasks)
        failures = [(int(r['case']), r['variant']) for r in rows if int(r['success']) == 0]
        self.assertEqual(failures, [(129, 'no-P')])

    def test_resume_rejects_code_change_even_when_complete(self):
        archived = control.ARCHIVED / 'null.csv'
        rows = control.read_rows(archived)
        tasks = [(r['suite'], int(r['N']), int(r['D']), int(r['case']), r['variant'])
                 for r in rows]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / 'null.csv').write_bytes(archived.read_bytes())
            manifest = out / 'null.manifest.json'
            # A missing manifest must never adopt pre-existing rows.
            args = ['runner', '--suite', 'null', '--output-dir', str(out)]
            with patch.object(sys, 'argv', args), patch.object(control, 'tasks_for', return_value=tasks):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, 'no source manifest'):
                        control.main()
                    manifest.write_text(json.dumps({'sources': {'old-code.py': 'old-hash'}}))
                    with self.assertRaisesRegex(ValueError, 'different code or geometry'):
                        control.main()
            self.assertEqual((out / 'null.csv').read_bytes(), archived.read_bytes())

    def test_archived_verification_is_read_only(self):
        path = control.ARCHIVED / 'null.csv'
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in control.ARCHIVED.rglob('*') if p.is_file()}
        rows = control.read_rows(path)
        tasks = [(r['suite'], int(r['N']), int(r['D']), int(r['case']), r['variant'])
                 for r in rows]
        args = ['runner', '--suite', 'null', '--verify-archived']
        with patch.object(sys, 'argv', args), patch.object(control, 'tasks_for', return_value=tasks):
            with contextlib.redirect_stdout(io.StringIO()):
                control.main()
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})


if __name__ == '__main__':
    unittest.main()
