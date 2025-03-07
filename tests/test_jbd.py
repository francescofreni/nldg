"""
Tests for `nldg.jbd`.
"""

import numpy as np
from nldg.utils.uwedge import uwedge
from scipy.linalg import block_diag
from scipy.stats import ortho_group
from nldg.utils.jbd import jbd, ajbd


def test_jbd() -> None:
    """
    Test the functions jbd, uwedge and ajbd.
    """
    np.random.seed(1)
    p = 10
    m = 10
    M_list = np.zeros((m, p, p))
    threshold = 0.07
    Q = ortho_group.rvs(dim=p)

    for j in range(m):
        A = block_diag(
            np.random.rand(3, 3),
            np.random.rand(2, 2),
            np.random.rand(4, 4),
            np.random.rand(1, 1),
        )
        noise = 0.1 * np.random.rand(p, p)
        M_list[j, :, :] = Q.T @ A @ A.T @ Q + noise @ noise.T

    U, blocks, MBD_list = jbd(M_list, threshold)

    assert U.shape == (10, 10)
    assert MBD_list.shape == (10, 10, 10)
    assert len(blocks) == 4

    V, converged, iteration, meanoffdiag = uwedge(M_list)

    assert V.shape == (10, 10)

    U, blocks, MBD_list, t_opt, boff_opt = ajbd(M_list)

    assert U.shape == (10, 10)
    assert MBD_list.shape == (10, 10, 10)
    assert len(blocks) == 4
