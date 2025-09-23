from __future__ import annotations
from typing import Literal, Optional, Tuple
import importlib.util

import numpy as _np_cpu
import scipy.sparse as _sp_cpu

# Select how you want to process data, GPU if fully available, else CPU
_ON_GPU = False
_xp = _np_cpu
_sp = _sp_cpu
_cp = None
_cpx_sp = None

# GPU availability requires: cupy, cupyx.scipy.sparse, and cuSPARSE runtime
if importlib.util.find_spec("cupy") and importlib.util.find_spec("cupyx.scipy.sparse"):
    try:
        import cupy as _cp
        import cupyx.scipy.sparse as _cpx_sp
        from cupyx import cusparse as _cusparse  # force-load CUDA sparse runtime
        _ON_GPU = True
        _xp = _cp
        _sp = _cpx_sp
    except Exception:
        # cuSPARSE or other CUDA libs not present → stay on CPU
        _ON_GPU = False
        _xp = _np_cpu
        _sp = _sp_cpu

EPS = 1e-12

# Convert to sparse representation between CPU/GPU backends 
def _to_backend_csr(A) -> _sp.csr_matrix:
    """
    Convert an input sparse matrix to the active backend's CSR type:
      - GPU active: SciPy CSR -> cupyx CSR (device)
      - CPU active: cupyx CSR -> SciPy CSR (host)
      - Pass-through if already correct
    """
    if _ON_GPU:
        if isinstance(A, _cpx_sp.csr_matrix):
            return A
        if _sp_cpu.isspmatrix_csr(A):
            return _cpx_sp.csr_matrix(
                (_cp.asarray(A.data), _cp.asarray(A.indices), _cp.asarray(A.indptr)),
                shape=A.shape
            )
        if _sp_cpu.issparse(A):
            return _to_backend_csr(A.tocsr(copy=False))
        raise TypeError(f"Expected SciPy/cupyx sparse matrix, got {type(A)}")
    else:
        if _sp_cpu.isspmatrix_csr(A):
            return A
        # only attempt cupyx -> scipy if cupyx actually importable in this env
        if importlib.util.find_spec("cupyx"):
            try:
                if isinstance(A, _cpx_sp.csr_matrix):
                    return _sp_cpu.csr_matrix(
                        (_cp.asnumpy(A.data), _cp.asnumpy(A.indices), _cp.asnumpy(A.indptr)),
                        shape=A.shape
                    )
            except Exception:
                pass
        if hasattr(A, "tocsr"):
            return A.tocsr(copy=False)
        raise TypeError(f"Expected SciPy/cupyx sparse matrix, got {type(A)}")

# Core helpers 
def _require_csr(A, name: str):
    A = _to_backend_csr(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be square (got {A.shape})")
    if (A.data < -1e-12).any():
        raise ValueError(f"{name} contains negative values")
    return A

def _row_sums(A):
    rs = _xp.asarray(A.sum(axis=1)).ravel()
    rs[~_xp.isfinite(rs)] = 0.0
    return rs

def _row_normalize(A):
    inv = 1.0 / (_row_sums(A) + EPS)
    return _sp.diags(inv).dot(A)

def _symmetrize(A):
    return ((A + A.T) * 0.5).tocsr()

def _csr_row_entropy(A):
    """Row-wise Shannon entropy on the active backend (GPU/CPU)."""
    indptr, data = A.indptr, A.data
    n = A.shape[0]
    out = _xp.empty(n, dtype=_xp.float32)
    for i in range(n):
        seg = data[indptr[i]:indptr[i+1]]
        seg = _xp.asarray(seg)  # ensure backend array
        s = seg.sum()
        if s <= 0:
            out[i] = _xp.float32(0.0)
            continue
        p = seg / (s + EPS)
        out[i] = -_xp.sum(p * _xp.log(p + EPS), dtype=_xp.float32)
    return out

#  Public API
def compute_modality_weights(
    Ar,
    Aa,
    mode: Literal["entropy_inverse", "degree", "uniform"] = "entropy_inverse",
    temperature: float = 1.0,
):
    """
    Return per-cell weights (wr, wa) with wr+wa==1 on the active backend.

    Ar, Aa: CSR sparse matrices (SciPy or cupyx accepted)
    mode  : "entropy_inverse" | "degree" | "uniform"
    temperature: >1 softens toward 0.5; <1 sharpens.
    """
    Ar = _require_csr(Ar, "Ar")
    Aa = _require_csr(Aa, "Aa")
    if Ar.shape != Aa.shape:
        raise ValueError("Ar and Aa must have the same shape")
    n = Ar.shape[0]

    if mode == "uniform":
        wr = _xp.full(n, 0.5, dtype=_xp.float32)
        wa = 1.0 - wr
    elif mode == "degree":
        sr = _row_sums(Ar)
        sa = _row_sums(Aa)
        z = sr + sa + EPS
        wr, wa = (sr / z).astype(_xp.float32), (sa / z).astype(_xp.float32)
    elif mode == "entropy_inverse":
        Hr = _csr_row_entropy(Ar)
        Ha = _csr_row_entropy(Aa)
        sharp_r = 1.0 / (Hr + EPS)
        sharp_a = 1.0 / (Ha + EPS)
        z = sharp_r + sharp_a + EPS
        wr, wa = (sharp_r / z).astype(_xp.float32), (sharp_a / z).astype(_xp.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if temperature != 1.0:
        alpha = _xp.float32(_xp.clip(1.0 / float(temperature), 0.0, 1e6))
        wr, wa = wr**alpha, wa**alpha
        z = wr + wa + EPS
        wr, wa = (wr / z).astype(_xp.float32), (wa / z).astype(_xp.float32)

    wr = _xp.clip(wr, 0.0, 1.0)
    wa = _xp.clip(wa, 0.0, 1.0)
    z = wr + wa + EPS
    return (wr / z).astype(_xp.float32), (wa / z).astype(_xp.float32)

def fuse_adjacencies(
    Ar,
    Aa,
    wr,
    wa,
    add_self_loops: bool = True,
    self_loop_weight: float = 1e-3,
    symmetric: bool = True,
    row_normalize: bool = True,
    prune_below: Optional[float] = 1e-6,
):
    """
    A_fused = normalize( symmetrize( diag(wr) @ Ar + diag(wa) @ Aa ) + self-loops )
    """
    Ar = _require_csr(Ar, "Ar")
    Aa = _require_csr(Aa, "Aa")
    n = Ar.shape[0]
    if wr.shape[0] != n or wa.shape[0] != n:
        raise ValueError("wr/wa length must equal number of rows")

    A = _sp.diags(wr).dot(Ar) + _sp.diags(wa).dot(Aa)

    if add_self_loops and self_loop_weight > 0:
        A = A + _sp.eye(n, dtype=A.dtype, format="csr") * float(self_loop_weight)

    if symmetric:
        A = _symmetrize(A)

    A.data = _xp.maximum(A.data, 0.0)

    if row_normalize:
        A = _row_normalize(A)

    if prune_below is not None and prune_below > 0:
        A.data[A.data < prune_below] = 0.0
        A.eliminate_zeros()
        assert (_row_sums(A) > 0).all(), "pruning produced zero-degree rows"

    return A

def fuse_from_mudata(
    mdata: "MuData",
    rna_key: str = "rna",
    atac_key: str = "atac",
    *,
    weight_mode: Literal["entropy_inverse", "degree", "uniform"] = "entropy_inverse",
    temperature: float = 1.0,
    obsp_out: str = "wnn_connectivities",
    obs_weight_keys: Tuple[str, str] = ("wnn_weight_rna", "wnn_weight_atac"),
    ) -> None:
    """
    Pull per-modality graphs from `mdata.mod[*].obsp['connectivities']`,
    compute per-cell weights, fuse, and write back to `mdata`:
      - `mdata.obsp[obsp_out]` fused CSR (GPU CSR if GPU active, else SciPy CSR)
      - `mdata.obs[...]` per-cell weights (NumPy arrays for cross-compatibility)
    """
    from anndata import AnnData  # noqa: F401
    from muon import MuData      # noqa: F401

    ad_rna = mdata.mod[rna_key]
    ad_atac = mdata.mod[atac_key]
    if "connectivities" not in ad_rna.obsp or "connectivities" not in ad_atac.obsp:
        raise KeyError("Each modality must have obsp['connectivities'].")

    Ar = _require_csr(ad_rna.obsp["connectivities"], "Ar")
    Aa = _require_csr(ad_atac.obsp["connectivities"], "Aa")

    wr, wa = compute_modality_weights(Ar, Aa, mode=weight_mode, temperature=temperature)
    A = fuse_adjacencies(Ar, Aa, wr, wa,
                         add_self_loops=True, self_loop_weight=1e-3,
                         symmetric=True, row_normalize=True, prune_below=1e-6)

    mdata.obsp[obsp_out] = A
    if _ON_GPU:
        mdata.obs[obs_weight_keys[0]] = _cp.asnumpy(wr)
        mdata.obs[obs_weight_keys[1]] = _cp.asnumpy(wa)
    else:
        mdata.obs[obs_weight_keys[0]] = wr
        mdata.obs[obs_weight_keys[1]] = wa

# Utility for tests/benchmarks
def make_knn_graph_cpu(n=1000, k=30, seed=0) -> _sp_cpu.csr_matrix:
    rng = _np_cpu.random.default_rng(seed)
    rows = _np_cpu.repeat(_np_cpu.arange(n), k)
    cols = rng.integers(0, n, size=n*k)
    data = rng.random(n*k).astype(_np_cpu.float32)
    A = _sp_cpu.csr_matrix((data, (rows, cols)), shape=(n, n))
    rs = _np_cpu.asarray(A.sum(axis=1)).ravel() + EPS
    return _sp_cpu.diags(1.0/rs).dot(A)

def backend_name() -> str:
    return "GPU (CuPy + cuSPARSE)" if _ON_GPU else "CPU (NumPy/SciPy)"


