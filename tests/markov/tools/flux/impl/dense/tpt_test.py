# This file is part of scikit-time and MSMTools.
#
# Copyright (c) 2020, 2015, 2014 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time and MSMTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

r"""Unit test for the TPT-module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np
from sktime.markov.tools.util.birth_death_chain import BirthDeathChain
from tests.markov.tools.numeric import assert_allclose

from sktime.markov.tools.flux.dense import tpt


class TestTPT(unittest.TestCase):
    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.A = [0, 1]
        self.B = [8, 9]
        self.a = 1
        self.b = 8

        self.bdc = BirthDeathChain(q, p)
        self.T = self.bdc.transition_matrix()

        """Use precomputed mu, qminus, qplus"""
        self.mu = self.bdc.stationary_distribution()
        self.qplus = self.bdc.committor_forward(self.a, self.b)
        self.qminus = self.bdc.committor_backward(self.a, self.b)
        # self.qminus = committor.backward_committor(self.T, self.A, self.B, mu=self.mu)
        # self.qplus = committor.forward_committor(self.T, self.A, self.B)
        self.fluxn = tpt.flux_matrix(self.T, self.mu, self.qminus, self.qplus, netflux=False)
        self.netfluxn = tpt.to_netflux(self.fluxn)
        self.Fn = tpt.total_flux(self.fluxn, self.A)
        self.kn = tpt.rate(self.Fn, self.mu, self.qminus)

    def test_flux(self):
        flux = self.bdc.flux(self.a, self.b)
        assert_allclose(self.fluxn, flux)

    def test_netflux(self):
        netflux = self.bdc.netflux(self.a, self.b)
        assert_allclose(self.netfluxn, netflux)

    def test_totalflux(self):
        F = self.bdc.totalflux(self.a, self.b)
        assert_allclose(self.Fn, F)

    def test_rate(self):
        k = self.bdc.rate(self.a, self.b)
        assert_allclose(self.kn, k)


if __name__ == "__main__":
    unittest.main()
