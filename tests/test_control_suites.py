"""Focused regression checks for observation composition and new evidence.

Run: OPENBLAS_NUM_THREADS=1 python tests/test_control_suites.py
"""
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], '--num-agents', '200']
import Configurations as C
from controllers.MORPHICA import MORPHICA
from controllers.FiniteRelayMORPHICA import FiniteRelayMORPHICA


class EvidenceChecks(unittest.TestCase):
    def setUp(self):
        C._obs_packet_loss = 0
        C._obs_delay_steps = 0
        C._obs_position_noise = 0

    def test_diagnostic_damage_ignores_global_reseeding(self):
        import hashlib
        import json
        import random
        from tools.damage_mask import damage_mask
        random.seed(2)
        destroyed = damage_mask(1000, 500, 129)
        digest = hashlib.sha256(json.dumps(sorted(destroyed)).encode()).hexdigest()
        self.assertEqual(digest, '955fc0d75050cc4332195c2907c34637d6ba5b1e0c9d0fa17ec9e7d04a38fd8a')
        random.seed(800)
        self.assertEqual(damage_mask(1000, 500, 129), destroyed)

    def test_delay_keeps_noise_and_uses_old_position(self):
        C._obs_delay_steps = 1
        C._obs_position_noise = 5
        x = np.array([[0., 0., 0.], [40., 0., 0.], [80., 0., 0.]])
        a = MORPHICA(x, observation_seed=91)
        a.check_subnets(x, [0,1,2])
        current = x + np.array([100., 0., 0.])
        a.check_subnets(current, [0,1,2])
        rng = np.random.default_rng(91)
        rng.normal(0,5,size=x.shape)
        expected = x + rng.normal(0,5,size=x.shape)
        np.testing.assert_allclose(a.sensed_positions, expected)
        self.assertFalse(np.array_equal(a.sensed_positions, x))

    def test_seeded_contact_loss_is_undirected(self):
        C._obs_packet_loss = .5
        x = np.column_stack((np.arange(10)*3., np.zeros((10,2))))
        a,b = [MORPHICA(x, observation_seed=31) for _ in range(2)]
        a.check_subnets(x, list(range(10)));b.check_subnets(x,list(range(10)))
        np.testing.assert_array_equal(a.A_mat,a.A_mat.T)
        np.testing.assert_array_equal(a.A_mat,b.A_mat)
        self.assertFalse(np.diag(a.A_mat).any())

    def test_noise_changes_operator_geometry(self):
        C._obs_position_noise = 5
        x = np.array([[0.,0.,50.], [40.,0.,50.], [0.,40.,50.]])
        a = MORPHICA(x, anchor_region=np.array([500.,500.,50.]), observation_seed=72)
        a.check_subnets(x,[0,1,2])
        actual,_=a.F_scout(x,[0,1,2])
        # Passing a different true state must not change the sensed S readout.
        other=x.copy();other[0,:2]+=[200,300]
        comparison,_=a.F_scout(other,[0,1,2])
        np.testing.assert_array_equal(actual,comparison)
        actual,_=a.F_polarization(x)
        comparison,_=a.F_polarization(other)
        np.testing.assert_array_equal(actual,comparison)

    def test_flooding_stops_at_missing_edge(self):
        x=np.array([[0.,0.,50.],[90.,0.,50.],[180.,0.,50.],[600.,0.,50.]])
        a=FiniteRelayMORPHICA(x,relay_rounds=1)
        a.check_subnets(x,list(range(4)))
        np.testing.assert_array_equal(a.relay_reach[0], [True,True,False,False])
        b=FiniteRelayMORPHICA(x,relay_rounds=8)
        b.check_subnets(x,list(range(4)))
        np.testing.assert_array_equal(b.relay_reach[0], [True,True,True,False])
        np.testing.assert_array_equal(b.relay_reach[3], [False,False,False,True])

    def test_finite_rounds_recover_exact_aggregate_readout(self):
        x=np.array([[0.,0.,50.],[90.,10.,50.],[180.,30.,50.]])
        anchor=np.array([500.,500.,50.])
        a=MORPHICA(x,anchor_region=anchor)
        b=FiniteRelayMORPHICA(x,anchor_region=anchor,relay_rounds=3)
        sa,_=a.group_fusion_step(x,[0,1,2]);sb,_=b.group_fusion_step(x,[0,1,2])
        np.testing.assert_allclose(sa,sb,rtol=1e-12,atol=1e-12)

    def test_sparse_component_split_matches_previous_order(self):
        import Utils
        rng=np.random.default_rng(16)
        x=rng.uniform(0,400,size=(45,3));a=MORPHICA(x)
        a.check_subnets(x,list(range(45)))
        previous,idx=Utils.split_the_positions_into_clusters_and_indexes(x,a.num_of_subnets,a.A_mat.astype(float))
        self.assertEqual(a.index_of_subnets,idx)
        for old,new in zip(previous,a.positions_of_subnets):np.testing.assert_array_equal(old,new)

    def test_symmetric_equilibrium_and_localized_polar_order(self):
        anchor=np.array([500.,500.,50.]);x=np.tile(anchor,(9,1))
        theta=2*np.pi*np.arange(8)/8
        x[1:,:2]+=150*np.column_stack((np.cos(theta),np.sin(theta)))
        a=MORPHICA(x,anchor_region=anchor,ablation='no-P')
        b=MORPHICA(x,anchor_region=anchor)
        sa,_=a.group_fusion_step(x,list(range(9)));sb,_=b.group_fusion_step(x,list(range(9)))
        self.assertEqual(a.num_of_subnets,2)
        self.assertLess(np.linalg.norm(sa,axis=1).max(),1e-9)
        self.assertEqual(int((np.linalg.norm(sb[1:],axis=1)>.5).sum()),1)
        self.assertAlmostEqual(np.linalg.norm(sb[1:].mean(axis=0)),1/8)

    def test_capped_derivative_potential(self):
        cr=120.;r=200.;h=1e-3
        psi=lambda x:.5*x*x if x<=cr else cr*x-.5*cr*cr
        self.assertAlmostEqual((psi(r+h)-psi(r-h))/(2*h),cr,places=6)


if __name__=='__main__':
    sys.argv=[sys.argv[0]]
    unittest.main()
