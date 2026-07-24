"""Native DPT branch detection: a faithful port of scanpy's Haghverdi 2016 algorithm.

Branch detection is graph/label logic, not tensor algebra, so it lives in the binding
layer rather than the Rust core. It reads the diffusion map scrust already computed
(`obsm["X_diffmap"]`, `uns["diffmap_evals"]`) plus the pseudotime, builds the Haghverdi
DPT distance matrix, and recursively splits segments by the Kendall-tau correlation of the
distances to a segment's tips. The result is the partition scanpy writes to
`obs["dpt_groups"]`; the labels are arbitrary, so parity is measured by adjusted Rand index.

Ported from `scanpy/tools/_dpt.py` (`detect_branchings`, `select_segment`,
`detect_branching`, `_detect_branching`, `__detect_branching_haghverdi16`,
`kendall_tau_split`) and `scanpy/neighbors` (`_get_dpt_row`). Only the partition is built;
the segment-adjacency tree scanpy also computes is not part of `dpt_groups`. The one shared
external numeric is `scipy.stats.kendalltau`, which scanpy itself uses for each split's
initial tau — using the same routine is what keeps the split points identical.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

# scanpy/scrust treat an eigenvalue at or above this as the stationary state, weighted 1
# in the DPT distance rather than lambda / (1 - lambda). Matches diffusion.rs.
_STATIONARY_EIGENVALUE = 0.9994
_MIN_KENDALL_LENGTH = 5


def dpt_groups(
    adata,
    *,
    n_branchings: int,
    min_group_size: float = 0.01,
    allow_kendall_tau_shift: bool = True,
    n_dcs: int = 10,
) -> np.ndarray:
    """Integer branch label per cell, as scanpy's `dpt(n_branchings>0)` writes to obs."""
    eigen_basis = np.asarray(adata.obsm["X_diffmap"], dtype=np.float64)[:, :n_dcs]
    eigen_values = np.asarray(adata.uns["diffmap_evals"], dtype=np.float64)[:n_dcs]
    pseudotime = np.asarray(adata.obs["dpt_pseudotime"], dtype=np.float64)
    iroot = int(adata.uns["iroot"])
    distances = _dpt_distances(eigen_basis, eigen_values)

    brancher = _Brancher(
        distances=distances,
        pseudotime=pseudotime,
        iroot=iroot,
        n_branchings=n_branchings,
        min_group_size=int(min_group_size * distances.shape[0])
        if min_group_size < 1
        else int(min_group_size),
        allow_kendall_tau_shift=allow_kendall_tau_shift,
    )
    return brancher.detect_branchings()


def _dpt_distances(eigen_basis: np.ndarray, eigen_values: np.ndarray) -> np.ndarray:
    """The Haghverdi DPT distance matrix, from `scanpy.neighbors._get_dpt_row`.

    `distance[i, j]^2 = sum_k w_k (psi[i, k] - psi[j, k])^2`, with `w_k = (l_k/(1-l_k))^2`
    for a non-stationary eigenvalue and `1` for the stationary one. Building the scaled
    coordinates and taking pairwise Euclidean distances is exactly that sum under a sqrt.
    """
    scale = np.ones_like(eigen_values)
    transient = eigen_values < _STATIONARY_EIGENVALUE
    scale[transient] = eigen_values[transient] / (1.0 - eigen_values[transient])
    scaled = eigen_basis * scale[None, :]
    squared_norm = np.einsum("ij,ij->i", scaled, scaled)
    gram = scaled @ scaled.T
    squared = squared_norm[:, None] + squared_norm[None, :] - 2.0 * gram
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared)


class _Brancher:
    """scanpy's `DPT` branch detection, reduced to the segment partition it produces."""

    def __init__(
        self,
        *,
        distances: np.ndarray,
        pseudotime: np.ndarray,
        iroot: int,
        n_branchings: int,
        min_group_size: int,
        allow_kendall_tau_shift: bool,
    ) -> None:
        self.distances_dpt = distances
        self.pseudotime = pseudotime
        self.iroot = iroot
        self.n_branchings = n_branchings
        self.min_group_size = min_group_size
        self.allow_kendall_tau_shift = allow_kendall_tau_shift
        self.choose_largest_segment = False
        self.n_obs = distances.shape[0]

    def detect_branchings(self) -> np.ndarray:
        """Recursively split, then flatten the segments into a per-cell label array."""
        indices_all = np.arange(self.n_obs, dtype=int)
        segs = [indices_all]
        tip_0 = int(np.argmax(self.distances_dpt[self.iroot]))
        tips_all = np.array([tip_0, int(np.argmax(self.distances_dpt[tip_0]))])
        segs_tips = [tips_all]
        segs_undecided = [True]

        for _ in range(self.n_branchings):
            iseg, tips3 = self.select_segment(segs, segs_tips, segs_undecided)
            if iseg == -1:
                break
            self.detect_branching(segs, segs_tips, segs_undecided, iseg, tips3)

        labels = np.zeros(self.n_obs, dtype=np.int64)
        for iseg, seg in enumerate(segs):
            labels[seg] = iseg
        return labels

    def select_segment(self, segs, segs_tips, segs_undecided):
        """The segment with the most distant third tip, and that tip triple."""
        scores_tips = np.zeros((len(segs), 4))
        allindices = np.arange(self.n_obs, dtype=int)
        for iseg, seg in enumerate(segs):
            if segs_tips[iseg][0] == -1:
                continue
            d_seg = self.distances_dpt[np.ix_(seg, seg)]
            third_maximizer = None
            if segs_undecided[iseg]:
                for jseg in range(len(segs)):
                    if jseg == iseg:
                        continue
                    for itip in range(2):
                        if (
                            self.distances_dpt[segs_tips[jseg][1], segs_tips[iseg][itip]]
                            < 0.5
                            * self.distances_dpt[segs_tips[iseg][~itip], segs_tips[iseg][itip]]
                        ):
                            third_maximizer = itip
            tips = [int(np.where(allindices[seg] == tip)[0][0]) for tip in segs_tips[iseg]]
            dseg = d_seg[tips[0]] + d_seg[tips[1]]
            if not np.isfinite(dseg).any():
                continue
            third_tip = int(np.argmax(dseg))
            if third_maximizer is not None:
                dseg = dseg + d_seg[third_tip]
                fourth_tip = int(np.argmax(dseg))
                if fourth_tip != tips[0] and fourth_tip != third_tip:
                    tips[1] = fourth_tip
                    dseg = dseg - d_seg[tips[1]]
                else:
                    dseg = dseg - d_seg[third_tip]
            tips3 = np.append(tips, third_tip)
            score = dseg[tips3[2]] / d_seg[tips3[0], tips3[1]]
            score = len(seg) if self.choose_largest_segment else score
            if len(seg) <= self.min_group_size:
                score = 0
            scores_tips[iseg, 0] = score
            scores_tips[iseg, 1:] = tips3
        iseg = int(np.argmax(scores_tips[:, 0]))
        if scores_tips[iseg, 0] == 0:
            return -1, None
        return iseg, scores_tips[iseg, 1:].astype(int)

    def detect_branching(self, segs, segs_tips, segs_undecided, iseg, tips3) -> None:
        """Split segment `iseg` into sub-segments and update the segment lists in place."""
        seg = segs[iseg]
        d_seg = self.distances_dpt[np.ix_(seg, seg)]
        ssegs, ssegs_tips, trunk = self._detect_branching(d_seg, tips3)
        for inew, seg_new in enumerate(ssegs):
            ssegs[inew] = seg[seg_new]
            ssegs_tips[inew] = seg[ssegs_tips[inew]]
        segs.pop(iseg)
        segs_tips.pop(iseg)
        segs.insert(iseg, ssegs[trunk])
        segs_tips.insert(iseg, ssegs_tips[trunk])
        segs += [s for i, s in enumerate(ssegs) if i != trunk]
        segs_tips += [t for i, t in enumerate(ssegs_tips) if i != trunk]
        n_add = len(ssegs) - 1
        if len(ssegs) == 4:
            segs_undecided.pop(iseg)
            segs_undecided.insert(iseg, True)
        segs_undecided += [False for _ in range(n_add)]

    def _detect_branching(self, d_seg, tips):
        """Three tip-orderings, intersect them, and gather the undecided remainder."""
        single = self._detect_branching_single(d_seg, tips)
        masks = np.zeros((len(single), d_seg.shape[0]), dtype=bool)
        for iseg, seg in enumerate(single):
            masks[iseg][seg] = True
        nonunique = np.sum(masks, axis=0) > 1
        ssegs = []
        for mask in masks:
            mask[nonunique] = False
            ssegs.append(np.arange(d_seg.shape[0], dtype=int)[mask])
        ssegs_tips = []
        for inew, newseg in enumerate(ssegs):
            secondtip = newseg[np.argmax(d_seg[tips[inew]][newseg])]
            ssegs_tips.append(np.array([tips[inew], secondtip]))
        undecided = np.arange(d_seg.shape[0], dtype=int)[nonunique]
        if len(undecided) > 0:
            ssegs.append(undecided)
            tip_0 = undecided[np.argmax(d_seg[undecided[0]][undecided])]
            tip_1 = undecided[np.argmax(d_seg[tip_0][undecided])]
            ssegs_tips.append(np.array([tip_0, tip_1]))
            trunk = 3
        else:
            reference = [ssegs_tips[i][0] for i in range(3)]
            closest = np.zeros((3, 3), dtype=int)
            for i in range(3):
                for j in range(3):
                    if i != j:
                        closest[i, j] = ssegs[j][np.argmin(d_seg[reference[i]][ssegs[j]])]
            added = np.zeros(3)
            added[0] = d_seg[closest[1, 0], closest[0, 1]] + d_seg[closest[2, 0], closest[0, 2]]
            added[1] = d_seg[closest[0, 1], closest[1, 0]] + d_seg[closest[2, 1], closest[1, 2]]
            added[2] = d_seg[closest[1, 2], closest[2, 1]] + d_seg[closest[0, 2], closest[2, 0]]
            trunk = int(np.argmin(added))
        return ssegs, ssegs_tips, trunk

    def _detect_branching_single(self, d_seg, tips):
        """Split away each of the three tips in turn (Haghverdi 2016 flavour)."""
        orderings = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        return [self._branch_from_tip(d_seg, tips[p]) for p in orderings]

    def _branch_from_tip(self, d_seg, tips):
        """Cells split away from `tips[0]` at the Kendall-tau branching point."""
        idcs = np.argsort(d_seg[tips[0]])
        imax = self.kendall_tau_split(d_seg[tips[1]][idcs], d_seg[tips[2]][idcs])
        if imax > 0.95 * len(idcs) and self.allow_kendall_tau_shift:
            ibranch = int(0.95 * imax)
        else:
            ibranch = imax + 1
        return idcs[:ibranch]

    def kendall_tau_split(self, a: np.ndarray, b: np.ndarray) -> int:
        """The split index maximising kendalltau(a[:i], b[:i]) - kendalltau(a[i:], b[i:])."""
        n = a.size
        if n < 2 * _MIN_KENDALL_LENGTH + 2:
            return _MIN_KENDALL_LENGTH
        idx_range = np.arange(_MIN_KENDALL_LENGTH, n - _MIN_KENDALL_LENGTH - 1, dtype=int)
        corr_coeff = np.zeros(idx_range.size)
        pos_old = kendalltau(a[:_MIN_KENDALL_LENGTH], b[:_MIN_KENDALL_LENGTH])[0]
        neg_old = kendalltau(a[_MIN_KENDALL_LENGTH:], b[_MIN_KENDALL_LENGTH:])[0]
        for ii, i in enumerate(idx_range):
            diff_pos, diff_neg = self._kendall_tau_diff(a, b, i)
            pos = pos_old + self._kendall_tau_add(i, diff_pos, pos_old)
            neg = neg_old + self._kendall_tau_subtract(n - i, diff_neg, neg_old)
            pos_old, neg_old = pos, neg
            corr_coeff[ii] = pos - neg
        return _MIN_KENDALL_LENGTH + int(np.argmax(corr_coeff))

    @staticmethod
    def _kendall_tau_add(len_old: int, diff_pos: float, tau_old: float) -> float:
        return 2.0 / (len_old + 1) * (float(diff_pos) / len_old - tau_old)

    @staticmethod
    def _kendall_tau_subtract(len_old: int, diff_neg: float, tau_old: float) -> float:
        return 2.0 / (len_old - 2) * (-float(diff_neg) / (len_old - 1) + tau_old)

    @staticmethod
    def _kendall_tau_diff(a: np.ndarray, b: np.ndarray, i: int):
        a_pos = np.zeros(a[:i].size, dtype=int)
        a_pos[a[:i] > a[i]] = 1
        a_pos[a[:i] < a[i]] = -1
        b_pos = np.zeros(b[:i].size, dtype=int)
        b_pos[b[:i] > b[i]] = 1
        b_pos[b[:i] < b[i]] = -1
        diff_pos = float(np.dot(a_pos, b_pos))
        a_neg = np.zeros(a[i:].size, dtype=int)
        a_neg[a[i:] > a[i]] = 1
        a_neg[a[i:] < a[i]] = -1
        b_neg = np.zeros(b[i:].size, dtype=int)
        b_neg[b[i:] > b[i]] = 1
        b_neg[b[i:] < b[i]] = -1
        diff_neg = float(np.dot(a_neg, b_neg))
        return diff_pos, diff_neg
