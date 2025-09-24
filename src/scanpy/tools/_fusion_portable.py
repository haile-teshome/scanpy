from __future__ import annotations
from typing import Literal, Optional, Tuple
import importlib.util

import numpy as _np_cpu
import scipy.sparse as _sp_cpu

# -------------------- backend selection --------------------
_ON_GPU = False
_xp = _np_cpu
_sp = _sp_cpu
_cp = None
_cpx_sp = None

if importlib.util.find_spec("cupy") and importlib.util.find_spec("cupyx.scipy.sparse"):
    try:
        import cupy as _cp
        import cupyx.scipy.sparse as _cpx_sp
        from cupyx import cusparse as _cusparse  # noqa: F401
        _ = _cp.array([0]).sum()  # touch runtime
        _ON_GPU = True
        _xp = _cp
        _sp = _cpx_sp
    except Exception:
        _ON_GPU = False
        _xp = _np_cpu
        _sp = _sp_cpu

EPS = 1e-12

# -------------------- conversions & checks --------------------
def _to_backend_csr(A):
    """
    Convert to active backend CSR:
      - GPU: SciPy CSR -> cupyx CSR (device)
      - CPU: cupyx CSR -> SciPy CSR (host)
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
        if importlib.util.find_spec("cupy") and importlib.util.find_spec("cupyx.scipy.sparse"):
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

def _require_csr(A, name: str):
    A = _to_backend_csr(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be square (got {A.shape})")
    if (_xp.asarray(A.data) < -1e-12).any():
        raise ValueError(f"{name} contains negative values")
    return A

# -------------------- small ops --------------------
def _row_sums(A):
    rs = _xp.asarray(A.sum(axis=1)).ravel()
    rs[~_xp.isfinite(rs)] = 0.0
    return rs

def _symmetrize(A):
    return ((A + A.T) * 0.5).tocsr()

# ---------- fast CSR helpers (CPU & GPU) ----------
# cache row_ids by the CSR's indptr memory address to reuse across calls
_rid_cache: dict[int, object] = {}

def _indptr_addr(A) -> int:
    indptr = A.indptr
    if _ON_GPU:
        return int(indptr.data.ptr)  # CuPy device pointer
    else:
        return int(indptr.__array_interface__['data'][0])  # NumPy host pointer

def _csr_row_ids(A):
    """
    Return row_id for each nonzero in CSR matrix A (length = nnz).

    Fully device-side on CuPy using searchsorted (no host hop).
    Vectorized on NumPy as well.
    """
    key = _indptr_addr(A)
    cached = _rid_cache.get(key, None)
    if cached is not None and cached.shape[0] == A.nnz:
        return cached

    indptr = A.indptr  # shape (n+1,)
    nnz = A.nnz
    # rid[i] = largest r such that indptr[r] <= i < indptr[r+1]
    # i.e., searchsorted(indptr, i, 'right') - 1 for i in 0..nnz-1
    rid = _xp.searchsorted(indptr, _xp.arange(nnz, dtype=indptr.dtype), side='right') - 1
    # ensure 32-bit int for indexing data (works on both backends)
    rid = rid.astype(_xp.int32, copy=False)
    _rid_cache[key] = rid
    return rid

def _csr_row_scale_inplace(A, scale):
    """Scale each row i of CSR A by scale[i] in-place (no diag())."""
    rid = _csr_row_ids(A)
    A.data *= scale[rid]
    return A

def _row_normalize_inplace(A):
    """Row-normalize CSR A in-place (no diag())."""
    inv = 1.0 / (_row_sums(A) + EPS)
    _csr_row_scale_inplace(A, _xp.asarray(inv, dtype=A.data.dtype))
    return A

def _csr_row_entropy_vectorized(A):
    """
    Row-wise Shannon entropy without Python loops.
    H_i = -sum_j p_ij log(p_ij),  p_ij = a_ij / sum_j a_ij
    """
    n = A.shape[0]
    rs = _row_sums(A) + EPS
    rid = _csr_row_ids(A)
    p = A.data / rs[rid]
    term = p * _xp.log(p + EPS)
    out = _xp.zeros(n, dtype=_xp.float32)
    _xp.add.at(out, rid, term.astype(_xp.float32, copy=False))
    return -out

# -------------------- public API --------------------
def compute_modality_weights(
    Ar,
    Aa,
    mode: Literal["entropy_inverse", "degree", "uniform"] = "entropy_inverse",
    temperature: float = 1.0,
):
    """
    Return per-cell weights (wr, wa) with wr+wa==1 on the active backend.
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
        Hr = _csr_row_entropy_vectorized(Ar)
        Ha = _csr_row_entropy_vectorized(Aa)
        sharp_r = 1.0 / (Hr + EPS)
        sharp_a = 1.0 / (Ha + EPS)
        z = sharp_r + sharp_a + EPS
        wr, wa = (sharp_r / z).astype(_xp.float32), (sharp_a / z).astype(_xp.float32)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Temperature (avoid cp.clip on scalars)
    if temperature != 1.0:
        alpha = 1.0 / float(temperature)
        if alpha < 0.0: alpha = 0.0
        if alpha > 1e6: alpha = 1e6
        alpha = _xp.float32(alpha)
        wr, wa = wr**alpha, wa**alpha
        z = wr + wa + EPS
        wr, wa = (wr / z).astype(_xp.float32), (wa / z).astype(_xp.float32)

    # Clamp to [0,1] without xp.clip
    wr = _xp.minimum(_xp.maximum(wr, 0.0), 1.0)
    wa = _xp.minimum(_xp.maximum(wa, 0.0), 1.0)
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
    A_fused = normalize( symmetrize( rowScale(wr, Ar) + rowScale(wa, Aa) ) + self-loops )
    (Row scaling is done in-place on CSR data; no diag() materialization.)
    """
    Ar = _require_csr(Ar, "Ar")
    Aa = _require_csr(Aa, "Aa")
    n = Ar.shape[0]
    if wr.shape[0] != n or wa.shape[0] != n:
        raise ValueError("wr/wa length must equal number of rows")

    # Row-scale without diag()
    Ar_scaled = Ar.copy()
    Aa_scaled = Aa.copy()
    _csr_row_scale_inplace(Ar_scaled, _xp.asarray(wr, dtype=Ar_scaled.data.dtype))
    _csr_row_scale_inplace(Aa_scaled, _xp.asarray(wa, dtype=Aa_scaled.data.dtype))

    A = Ar_scaled + Aa_scaled

    if add_self_loops and self_loop_weight > 0:
        A = A + _sp.eye(n, dtype=A.dtype, format="csr") * float(self_loop_weight)

    if symmetric:
        A = _symmetrize(A)

    A.data = _xp.maximum(A.data, 0.0)

    if row_normalize:
        _row_normalize_inplace(A)

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
    compute per-cell weights, fuse, and write back to `mdata`.
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

# -------------------- small utility for tests --------------------
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


