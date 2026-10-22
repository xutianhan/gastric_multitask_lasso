"""Numerical regression checks independent of patient data and experiment files."""

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

for name in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from gastric_mtl.cox import c_index
from gastric_mtl.eac import prepare
from gastric_mtl.second_order import SmoothSharingObjective
from gastric_mtl.solver import StableDirectionalObjective, fit_stable_newton
from gastric_mtl.warmup import warmup
from gastric_mtl.integrity import definition_fingerprints


class NumericalRegressionTests(unittest.TestCase):
    def test_model_definitions_match_reviewed_fingerprints(self):
        """Check numerical definitions without keeping a second code version."""
        expected = json.loads((ROOT/'provenance/model_fingerprints.json').read_text())
        for current, fingerprint in expected.items():
            with self.subTest(module=current):
                self.assertEqual(definition_fingerprints(ROOT / current), fingerprint)

    def test_derivatives_and_complete_fits(self):
        """36 derivative cases and nine converged fits, with ties and extreme scales."""
        rng = np.random.RandomState(918)
        checks = fits = 0
        for count in [1, 2, 3]:
            for scale in [1e-5, .01, 1., 1000.]:
                for edge in [0., .001, 1.]:
                    with self.subTest(tasks=count, scale=scale, edge=edge):
                        tasks = [dict(X=rng.normal(size=(20+k, 15)),
                                      time=rng.randint(1, 8, size=20+k).astype(float),
                                      event=(rng.rand(20+k) < .6).astype(int)) for k in range(count)]
                        groups = np.repeat(np.arange(5), 3)
                        lam = np.linspace(.03, .1, count)
                        objective = StableDirectionalObjective(tasks, groups, lam, edge)
                        x = rng.normal(size=count*15) * scale
                        v = rng.normal(size=count*15)
                        v /= np.linalg.norm(v)
                        u = rng.normal(size=count*15)
                        step = min(1e-5, scale*.0001)
                        hv = objective.hessp(x, v)
                        fd = (objective.value_grad(x+step*v)[1] - objective.value_grad(x-step*v)[1])/(2*step)
                        error = np.max(np.abs(hv-fd))/max(1., np.max(np.abs(hv)))
                        symmetry = abs(u@hv-v@objective.hessp(x, u))/max(1., abs(u@hv))
                        self.assertLess(error, 2e-5)
                        self.assertLess(symmetry, 1e-9)
                        if scale < 1000:
                            reference = SmoothSharingObjective(tasks, groups, lam, edge)
                            a, g = reference.value_grad(x)
                            b, gg = objective.value_grad(x)
                            self.assertLess(max(abs(a-b), np.max(np.abs(g-gg)),
                                                np.max(np.abs(reference.hessp(x, v)-hv))), 1e-9)
                        checks += 1
                        if scale == .01:
                            initial = None
                            if count > 1:
                                pre = warmup(tasks, groups, lam, edge, np.zeros((count, 15)))
                                self.assertLessEqual(pre['objective'], pre['initial_objective']+1e-8)
                                initial = pre['betas']
                            fit = fit_stable_newton(tasks, groups, lam, edge, beta0=initial, max_iter=600)
                            self.assertTrue(fit['converged'])
                            self.assertLessEqual(fit['objective'], fit['initial_objective']+1e-8)
                            value, gradient = objective.value_grad(np.asarray(fit['betas']).ravel())
                            self.assertLess(abs(value-fit['objective']), 1e-10)
                            self.assertLessEqual(np.max(np.abs(gradient)), 1e-5)
                            fits += 1
        self.assertEqual((checks, fits), (36, 9))

    def test_zero_coupling_equals_independent_objectives(self):
        rng = np.random.RandomState(42)
        tasks = [dict(X=rng.randn(12, 6), time=np.arange(1, 13.), event=np.ones(12)) for _ in range(3)]
        groups = np.repeat(np.arange(2), 3)
        lam = [.03, .05, .1]
        coefficients = rng.randn(3, 6) * .1
        joint_value, joint_grad = StableDirectionalObjective(tasks, groups, lam, 0).value_grad(coefficients.ravel())
        singles = [StableDirectionalObjective([task], groups, [penalty], 0).value_grad(beta)
                   for task, penalty, beta in zip(tasks, lam, coefficients)]
        self.assertAlmostEqual(joint_value, sum(value for value, _ in singles), places=12)
        np.testing.assert_allclose(joint_grad, np.concatenate([grad for _, grad in singles]), atol=1e-12, rtol=0)

    def test_holdout_expression_cannot_change_training_state(self):
        rng = np.random.RandomState(10)
        cohorts = {c: dict(X_log2=rng.randn(10, 5), time=np.arange(1, 11.), event=np.ones(10))
                   for c in ['EAC', 'STAD']}
        train = [np.arange(7), np.arange(7)]
        test = [np.arange(7, 10), np.arange(7, 10)]
        labels = np.array([0, 0, 1, 1, 2])
        first = prepare(cohorts, train, test, labels)
        altered = copy.deepcopy(cohorts)
        for c in altered:
            altered[c]['X_log2'][7:] = 1e9
        second = prepare(altered, train, test, labels)
        np.testing.assert_array_equal(first[3], second[3])
        for a, b in zip(first[0], second[0]):
            np.testing.assert_array_equal(a['X'], b['X'])
            np.testing.assert_allclose(a['X'].mean(axis=0), 0, atol=1e-12)
            np.testing.assert_allclose(a['X'].std(axis=0), 1, atol=1e-12)
        for a, b in zip(first[4], second[4]):
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])

    def test_concordance_event_censor_ties_and_risk_ties(self):
        time = [1, 1, 2, 3]
        event = [1, 0, 1, 0]
        self.assertEqual(c_index(time, event, [3, 2, 1, 0]), 1.)
        self.assertEqual(c_index(time, event, [0, 1, 2, 3]), 0.)
        self.assertEqual(c_index(time, event, [0, 0, 0, 0]), .5)


if __name__ == '__main__':
    unittest.main()
