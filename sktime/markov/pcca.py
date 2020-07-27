# This file is part of scikit-time
#
# Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time is free software: you can redistribute it and/or modify
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


import warnings
from typing import List

import numpy as np

from sktime.base import Model
from sktime.numeric import mdot


# TODO: should pass pi to msmtools once it's supported.
def pcca(P, m, stationary_distribution=None):
    """PCCA+ spectral clustering method with optimized memberships.

    Implementation according to :cite:`pcca-roblitz2013fuzzy`.
    Clusters the first m eigenvectors of a transition matrix in order to cluster the states.
    This function does not assume that the transition matrix is fully connected. Disconnected sets
    will automatically define the first metastable states, with perfect membership assignments.

    Parameters
    ----------
    P : ndarray (n,n)
        Transition matrix.

    m : int
        Number of clusters to group to.

    stationary_distribution : ndarray(n,), optional, default=None
        Stationary distribution over the full state space, can be given if already computed.

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: pcca-
    """
    if m <= 0 or m > P.shape[0]:
        raise ValueError("Number of metastable sets must be larger than 0 and can be at most as large as the number "
                         "of states.")
    assert 0 < m <= P.shape[0]
    from scipy.sparse import issparse
    if issparse(P):
        warnings.warn('PCCA is only implemented for dense matrices, '
                      'converting sparse transition matrix to dense ndarray.', stacklevel=2)
        P = P.toarray()

    # stationary distribution
    if stationary_distribution is None:
        from sktime.markov.tools.analysis import stationary_distribution as statdist
        pi = statdist(P)
    else:
        pi = stationary_distribution

    # memberships
    # TODO: can be improved. pcca computes stationary distribution internally, we don't need to compute it twice.
    from sktime.markov.tools.analysis.dense.pcca import pcca as _algorithm_impl
    M = _algorithm_impl(P, m)

    # coarse-grained stationary distribution
    pi_coarse = np.dot(M.T, pi)

    # HMM output matrix
    B = mdot(np.diag(1.0 / pi_coarse), M.T, np.diag(pi))
    # renormalize B to make it row-stochastic
    B /= B.sum(axis=1)[:, None]

    # coarse-grained transition matrix
    W = np.linalg.inv(np.dot(M.T, M))
    A = np.dot(np.dot(M.T, P), M)
    P_coarse = np.dot(W, A)

    # symmetrize and renormalize to eliminate numerical errors
    X = np.dot(np.diag(pi_coarse), P_coarse)
    # and normalize
    P_coarse = X / X.sum(axis=1)[:, None]

    return PCCAModel(P_coarse, pi_coarse, M, B)


class PCCAModel(Model):
    """
    Model for PCCA+ spectral clustering method with optimized memberships.

    PCCA+ spectral clustering is described in :cite:`pccamodel-roblitz2013fuzzy`.
    Clusters the first m eigenvectors of a transition matrix in order to cluster the states.
    This function does not assume that the transition matrix is fully connected. Disconnected sets
    will automatically define the first metastable states, with perfect membership assignments.

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: pccamodel-
    """

    def __init__(self, transition_matrix_coarse: np.ndarray, pi_coarse: np.ndarray, memberships: np.ndarray,
                 metastable_distributions: np.ndarray):
        r""" Creates a new model instance.

        Parameters
        ----------
        transition_matrix_coarse : ndarray (n,n)
            Coarse transition matrix.
        pi_coarse : ndarray (n,)
            Coarse stationary distribution
        memberships : ndarray (n,m)
            The pcca memberships to clusters
        metastable_distributions : ndarray (m, n)
            metastable distributions
        """
        super().__init__()
        self._transition_matrix_coarse = transition_matrix_coarse
        self._pi_coarse = pi_coarse
        self._memberships = memberships
        self._metastable_distributions = metastable_distributions
        self._m = self._memberships.shape[1]

    @property
    def n_metastable(self):
        r""" Number of metastable states. """
        return self._m

    @property
    def memberships(self):
        r""" Probabilities of MarkovStateModel states to belong to a metastable state by PCCA+

        Returns the memberships of active set states to metastable sets.

        Returns
        -------
        M : ndarray((n,m))
            A matrix containing the probability or membership of each state to be
            assigned to each metastable set, i.e. p(metastable | state).
            The row sums of M are 1.
        """
        return self._memberships

    @property
    def metastable_distributions(self):
        r""" Probability of metastable states to visit an MarkovStateModel state by PCCA+

        Returns the probability distributions of active set states within
        each metastable set by combining the PCCA+ method with
        Bayesian inversion as described in :cite:`pccamodel-noe2013projected`.

        Returns
        -------
        p_out : ndarray (m,n)
            A matrix containing the probability distribution of each active set
            state, given that we are in one of the m metastable sets,
            i.e. p(state | metastable). The row sums of p_out are 1.
        """
        return self._metastable_distributions

    @property
    def coarse_grained_transition_matrix(self):
        r""" Coarse grained transition matrix with :attr:`n_metastable` states. """
        return self._transition_matrix_coarse

    @property
    def coarse_grained_stationary_probability(self):
        r""" Stationary distribution for :attr:`coarse_grained_transition_matrix`. """
        return self._pi_coarse

    @property
    def assignments(self) -> np.ndarray:
        """ Assignment of states to metastable sets using PCCA++

        Computes the assignment to metastable sets for active set states using
        the PCCA++ method :cite:`pccamodel-roblitz2013fuzzy`.

        This is only recommended for visualization purposes. You *cannot* compute
        any actual quantity of the coarse-grained kinetics without employing the
        fuzzy memberships!

        Returns
        -------
        assignments : ndarray (n,)
            For each MarkovStateModel state, the metastable state it is located in.

        """
        return np.argmax(self.memberships, axis=1)

    @property
    def sets(self) -> List[np.ndarray]:
        """ Metastable sets using PCCA+

        Computes the metastable sets of active set states within each
        metastable set using the PCCA+ method :cite:`pccamodel-roblitz2013fuzzy`.

        This is only recommended for visualization purposes. You *cannot*
        compute any actual quantity of the coarse-grained kinetics without
        employing the fuzzy memberships!

        Returns
        -------
        sets : list of ndarray
            A list of length equal to metastable states. Each element is an
            array with microstate indexes contained in it
        """
        res = []
        assignment = self.assignments
        for i in range(self.n_metastable):
            res.append(np.where(assignment == i)[0])
        return res
