"""
Code adapted from https://github.com/mlazzaretto/Invariant-Subspace-Decomposition.git
Original Author: Margherita Lazzaretto
"""

from __future__ import division
import numpy as np
from nldg.uwedge import uwedge


def jbd(
    M_list: np.ndarray,
    threshold: float,
    diag: bool = False,
) -> tuple[np.ndarray, list, np.ndarray] | tuple[np.ndarray, bool, int, float]:
    """
    Joint block diagonalization of the input set given a threshold.

    Args:
        M_list: list of matrices to be jointly block diagonalized.
        threshold: mean off-diagonal elements approx. zero if < threshold.
        diag: if True, approximate joint diagonalization is computed.

    Returns:
        A tuple of 2 elements:
        - U: joint block diagonalizer.
        - blocks: list of blocks as permutation from uwedge.
        - MBD_list: list of jointly block diagonalized matrices.
    """
    m = M_list.shape[0]
    p = M_list.shape[1]

    # Approximate joint diagonalization
    V, converged, iteration, meanoffdiag = uwedge(M_list)
    if diag:
        return V, converged, iteration, meanoffdiag

    MD_list = np.zeros_like(M_list)
    for k in range(m):
        MD_list[k, :, :] = V@M_list[k, :, :]@V.T.conj()

    # Compute the auxiliary matrix
    B = np.max(np.abs(MD_list), axis=0)
    B = B / np.linalg.norm(B)

    # Detect blocks according to threshold
    U = V.copy()

    blocks = [1, ]
    last = 0
    for j in range(p-1):
        b = B.copy()[:, j]
        b_ord = list(np.argsort(-b[last+1:]) + last + 1)
        while len(b_ord) > 0 and b[b_ord[0]] > threshold:
            blocks[-1] += 1
            last += 1
            # Swap rows and columns
            B[[last, b_ord[0]], :] = B[[b_ord[0], last], :]
            B[:, [last, b_ord[0]]] = B[:, [b_ord[0], last]]
            U[[last, b_ord[0]], :] = U[[b_ord[0], last], :]
            b = B.copy()[:, j]
            b_ord = list(np.argsort(-b[last+1:]) + last + 1)
        if j == last:
            blocks.append(1)
            last += 1

    # Compute jointly block diagonalized matrices
    MBD_list = np.zeros_like(M_list)
    for k in range(m):
        MBD_list[k, :, :] = U@M_list[k, :, :]@U.T.conj()

    return U, blocks, MBD_list


def boff(
    MBD_list: np.ndarray,
    blocks_shape: list,
) -> float:
    """
    Compute penalized mean of the off-block-diagonal elements of a set of matrices.

    Args:
        MBD_list: list of jointly block diagonalized matrices.
        blocks: set of blocks for the input matrices.

    Returns:
        crit: penalized mean off-block-diagonal value.
    """
    m = MBD_list.shape[0]
    p = MBD_list.shape[1]

    # Compute mean off-block-diagonal value
    MOD_list = np.copy(MBD_list)
    C = np.zeros(MBD_list.shape[0])
    # blocks_shape = [len(block) for block in blocks]

    if len(blocks_shape) == 1:
        for j in range(MOD_list.shape[0]):
            C[j] = np.linalg.norm(MOD_list[j, :, :], 'fro')
        boff_crit = sum(C)/(m)
    else:
        for b, bs in enumerate(blocks_shape):
            if b == 0:
                block_idxs = list(range(bs))
            else:
                block_idxs = [j+sum(blocks_shape[:b]) for j in range(bs)]
            MOD_list[:, block_idxs[0]:block_idxs[-1]+1,
                     block_idxs[0]:block_idxs[-1]+1] = 0

        for j in range(m):
            C[j] = np.linalg.norm(MOD_list[j, :, :], 1)

        boff_crit = sum(C)/(m*(p**2-sum([bs**2 for bs in blocks_shape])))

    # Compute penalization term
    lam = np.mean([np.min(np.linalg.eigvals(Sigma)) for Sigma in MBD_list])

    # Evaluate joint criterion
    crit = boff_crit + lam*sum([bs**2 for bs in blocks_shape])/(p**2)

    return crit


def ajbd(
    M_list: np.array,
    diag: bool = False,
) -> (tuple[np.ndarray, list, np.ndarray, float, float] |
      tuple[np.ndarray, list, np.ndarray] |
      tuple[np.ndarray, bool, int, float]):
    """
    Compute approximate joint block diagonalization of the set of input matrices.

    Args:
        M_list: set of matrices to be jointly block diagonalized.
        diag: if True, approximate joint diagonalization is computed.

    Returns:
        A tuple of 5 elements:
        - U: joint block diagonalizer.
        - blocks_list[idx]: list of dimensions of the detected blocks.
        - MBD_list: list of jointly block diagonalized matrices.
        - t_opt: optimal threshold for jbd.
        - boff_list[idx]: optimal objective value.
    """
    # Return joint block diagonalization if diag=True
    if diag:
        return jbd(M_list, 0, diag)

    m = M_list.shape[0]
    # Initialize list of possible thresholds
    t_list = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

    # Evaluate mean off-block-diagonal for all thresholds
    boff_list = []
    blocks_list = []
    U_list = []
    for t in t_list:
        U, blocks, MBD_list = jbd(M_list, t)
        boff_list.append(boff(MBD_list, blocks))
        blocks_list.append(blocks)
        U_list.append(U)

    # Select optimal threshold
    idx = np.argmin(np.array(boff_list))
    U = U_list[idx]
    MBD_list = np.zeros_like(M_list)
    for k in range(m):
        MBD_list[k, :, :] = U@M_list[k, :, :]@U.T.conj()
    t_opt = t_list[idx]

    return U, blocks_list[idx], MBD_list, t_opt, boff_list[idx]
