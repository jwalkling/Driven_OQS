"""
Lindblad utilities
================================

Core library for constructing and analysing open quantum systems
described within the Lindblad master equation formalism.

This module provides:

- Pauli matrices and jump operators.
- Lindblad model and Liouvillian builder classes.
- Trajectory and evolution classes for simulating the dynamics of open quantum systems.
- Biorthonormal eigensystem computation for Liouvillian operators.
"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



#-----------------------------------
# Classes for Lindblad dynamics
#-----------------------------------


#Pauli matrices
class Pauli:
    sx = np.array([[0, 1],[1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0],[0, -1]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
    sp = np.array([[0, 1],[0, 0]], dtype=np.complex128)   # σ+
    sm = np.array([[0, 0],[1, 0]], dtype=np.complex128)   # σ-


#Liouvillian operators and builders
class Jump:
    def __init__(self, L: np.ndarray, rate: float):
        self.L = np.asarray(L, dtype=np.complex128)
        self.rate = float(rate)

class LindbladModel2LS:
    """
    Defines a time-dependent 2LS Lindblad model:
      H(t): 2x2
      jumps(t): list[Jump]
    """
    def __init__(self, H_of_t, jumps_of_t):
        self.H_of_t = H_of_t
        self.jumps_of_t = jumps_of_t

    def H(self, t: float) -> np.ndarray:
        return np.asarray(self.H_of_t(t), dtype=np.complex128)

    def jumps(self, t: float) -> list[Jump]:
        return list(self.jumps_of_t(t))
    

class LiouvillianBuilder:
    """
    Builds L such that d/dt vec(rho) = L vec(rho).
    Convention used:
    vec(AXB) = (B^T ⊗ A) vec(X)
    and vec uses order='F' when reshaping matrices <-> vectors.
    """
    def __init__(self, dim: int = 2, vec_order: str = "F"):
        print(dim)
        self.d = int(dim)

        self.vec_order = vec_order
        self.I = np.eye(self.d, dtype=np.complex128)

    def vec(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.complex128).reshape(self.d * self.d, order=self.vec_order)

    def mat(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.complex128).reshape(self.d, self.d, order=self.vec_order)

    def build(self, H: np.ndarray, jumps: list) -> np.ndarray:
        H = np.asarray(H, dtype=np.complex128)
        I = self.I

        # -i[H, ρ]
        L = -1j * (np.kron(I, H) - np.kron(H.T, I))

        for j in jumps:
            gamma = j.rate
            if gamma == 0.0:
                continue

            Lk = j.L
            LdL = Lk.conj().T @ Lk

            term_jump  = np.kron(Lk.conj(), Lk)       # L ρ L†
            term_left  = -0.5 * np.kron(I, LdL)       # -1/2 (L†L) ρ
            term_right = -0.5 * np.kron(LdL.T, I)     # -1/2 ρ (L†L)

            L += gamma * (term_jump + term_left + term_right)

        return L #note that this L is in the acts on vec(rho) space. Will not be the same as in the literature where we have the Bloch form.


# Evolution and Trajectory classes
class Trajectory:
    def __init__(self, t: np.ndarray, rho_vec: np.ndarray, builder):
        self.t = np.asarray(t, dtype=float)
        self.rho_vec = np.asarray(rho_vec, dtype=np.complex128)  # (nt, d^2)
        self._builder = builder

    def rho(self, k: int) -> np.ndarray:
        return self._builder.mat(self.rho_vec[k])

class LindbladEvolver:
    def __init__(self, model, builder):
        self.model = model
        self.builder = builder

    def simulate(self, rho0: np.ndarray, t_eval: np.ndarray, rtol=1e-9, atol=1e-12) -> Trajectory:
        rho0 = np.asarray(rho0, dtype=np.complex128)
        y0 = self.builder.vec(rho0)
        t_span = (float(t_eval[0]), float(t_eval[-1]))

        def rhs(t, y):
            H = self.model.H(t)
            jumps = self.model.jumps(t)
            L = self.builder.build(H, jumps)
            return L @ y

        sol = solve_ivp(rhs, t_span, y0, t_eval=np.asarray(t_eval, dtype=float),
                        rtol=rtol, atol=atol)

        rho_vec = sol.y.T  # (nt, d^2)
        return Trajectory(sol.t, rho_vec, self.builder)


class Biorth:
    """Biorthonormal eigensystem for L: L R = R Λ and L† W = W Λ* with W†R = I (when diagonalizable)."""
    def __init__(self):
        self.evals = None
        self.R = None
        self.W = None
        self.is_defective = False
        self.info = {}

    @staticmethod
    def compute(
        L: np.ndarray,
        *,
        tol_zero_real: float = 1e-12,
        rcond: float = 1e-12,
        allow_pinv: bool = True,
        raise_on_defective: bool = False,
    ):
        evals, R = np.linalg.eig(L)
        evalsL, V = np.linalg.eig(L.conj().T)

        # match left eigenvectors to conjugate right eigenvalues (greedy)
        W = np.zeros_like(R)
        used = np.zeros(len(evalsL), dtype=bool)
        for i, lam in enumerate(evals):
            # pick nearest not-yet-used (helps a bit with degeneracies)
            diffs = np.abs(evalsL - lam.conjugate())
            diffs = np.where(used, np.inf, diffs)
            j = int(np.argmin(diffs))
            used[j] = True
            W[:, i] = V[:, j]

        # Attempt biorthonormalization: solve W† (R X) = I  => (W†R) X = I
        M = W.conj().T @ R

        # Diagnose defectiveness / ill-conditioning
        s = np.linalg.svd(M, compute_uv=False)
        # rank estimate
        rank = int(np.sum(s > rcond * s[0])) if s.size and s[0] > 0 else 0
        cond = (s[0] / s[-1]) if (s.size and s[-1] > 0) else np.inf

        is_defective = (rank < M.shape[0]) or (not np.isfinite(cond)) or (cond > 1.0 / rcond)

        if is_defective and raise_on_defective:
            raise np.linalg.LinAlgError(
                "Biorthonormal eigenbasis does not exist / is ill-conditioned (defective or near-defective matrix). "
                "Need generalized eigenvectors (Jordan chains) instead."
            )

        if is_defective and not allow_pinv:
            # return raw (will not be biorthonormal) but at least flagged
            Rn = R
        else:
            Minv = np.linalg.pinv(M, rcond=rcond) if is_defective else np.linalg.inv(M)
            Rn = R @ Minv

        # ---- sort by real part: largest (≈0) to smallest (negative) ----
        re = evals.real
        im = evals.imag
        re0 = np.where(np.abs(re) < tol_zero_real, 0.0, re)
        order = np.lexsort((-im, np.abs(re0), -re0))

        evals = evals[order]
        Rn = Rn[:, order]
        W = W[:, order]

        out = Biorth()
        out.evals, out.R, out.W = evals, Rn, W
        out.is_defective = bool(is_defective)
        out.info = {"rank_M": rank, "cond_M": cond, "singvals_M": s}
        return out
    

def pure_rho(psi: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Convert a state vector |psi> to density matrix rho = |psi><psi|.
    psi: shape (2,) (or (d,))
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)

    if normalize:
        n = np.vdot(psi, psi)  # <psi|psi>
        if n == 0:
            raise ValueError("psi has zero norm.")
        psi = psi / np.sqrt(n)

    return np.outer(psi, psi.conj())

def trace(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho)))

def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))

#--------------------------------
# Conversion from Bloch to other forms
#--------------------------------

def bloch(rho: np.ndarray) -> np.ndarray:
    return np.array([
        np.real(np.trace(rho @ Pauli.sx)),
        np.real(np.trace(rho @ Pauli.sy)),
        np.real(np.trace(rho @ Pauli.sz)),
    ], dtype=float)

def supervec_rho_to_bloch(v, *, check_shape=True):
    """
    Convert a 4-supervector v = (rho00, rho01, rho10, rho11)^T
    to Bloch/Pauli coefficients b = (bI, bx, by, bz)^T where

        rho = 0.5 * (bI*I + bx*sx + by*sy + bz*sz).

    Notes:
      - Uses basis order [I, sx, sy, sz] (unnormalized Pauli basis).
      - Assumes v corresponds to the matrix entries, not Liouville-space basis states.
    """
    v = np.asarray(v)
    if check_shape and v.shape[-1] != 4:
        raise ValueError(f"Expected last dimension 4, got {v.shape}")

    rho00, rho01, rho10, rho11 = np.moveaxis(v, -1, 0)

    bI = rho00 + rho11
    bx = rho01 + rho10
    by = 1j * (rho01 - rho10)   # = Tr(sy rho)
    bz = rho00 - rho11

    return np.stack([bI, bx, by, bz], axis=-1)

def bloch_to_operator(v):
    """
    Build the operator v_x σx + v_y σy + v_z σz from a 3-vector v.
    v can be length-3 array-like.
    Returns a 2x2 complex ndarray.
    """
    v = np.asarray(v, dtype=float).ravel()
    if v.size != 3:
        raise ValueError("Expected v of length 3 (bx,by,bz).")
    return v[0] * Pauli.sx + v[1] * Pauli.sy + v[2] * Pauli.sz


#------------------------------------
# Steady state calculations
#------------------------------------

def get_steady_state(builder, model, t: float, *, normalize_trace: bool = True, return_eval: bool = False):
    """
    Compute the instantaneous steady-state density matrix for the Lindbladian
    L(t) = builder.build(model.H(t), model.jumps(t)).

    Strategy:
      - diagonalise L(t)
      - pick the eigenvector whose eigenvalue has smallest absolute value
        (closest to zero steady-state)
      - reshape to matrix via builder.mat and optionally normalise trace -> 1

    Parameters
    ----------
    builder : LiouvillianBuilder
        Builder used to vectorise/unvectorise states and to build L.
    model : LindbladModel2LS (or compatible object)
        Must provide model.H(t) and model.jumps(t).
    t : float
        Time at which to evaluate the instantaneous steady state.
    normalize_trace : bool, optional
        If True (default) normalise returned rho so trace(rho)=1 when possible.
    return_eval : bool, optional
        If True also return the eigenvalue associated with the returned eigenvector.

    Returns
    -------
    rho_ss : ndarray (d,d)
        Instantaneous steady-state density matrix (not guaranteed strictly positive
        due to numerical error).
    eval_ss : complex, optional
        If return_eval is True, also return the corresponding eigenvalue.
    """
    # Build Liouvillian at time t
    L = builder.build(model.H(t), model.jumps(t))

    # Diagonalise
    evals, R = np.linalg.eig(L)

    # pick eigenvector with eigenvalue closest to zero
    idx = int(np.argmin(np.abs(evals)))
    vec = R[:, idx]

    # reshape to matrix (builder.mat uses vec_order set on the builder)
    rho_ss = builder.mat(vec)

    # normalise trace if requested and possible
    if normalize_trace:
        tr = np.trace(rho_ss)
        if np.abs(tr) > 0:
            rho_ss = rho_ss / tr

    if return_eval:
        return rho_ss, evals[idx]
    return rho_ss

def get_steady_state_zz_of_t(builder, model, t: np.ndarray):
    """
    Compute the instantaneous steady-state prediction for <σz> at times t.
    """
    sz_ss_t = np.zeros_like(t, dtype=float)
    for i, tt in enumerate(t):
        rho_ss = get_steady_state(builder, model, tt)
        sz_ss_t[i] = np.real(np.trace(rho_ss @ Pauli.sz))
    return sz_ss_t


def get_steady_state_blochvector_of_t(builder, model, t):
    """
    Compute the instantaneous steady-state Bloch vector [bx, by, bz] at times t.

    Returns:
      b_t : np.ndarray shaped (nt, 3) with columns [bx, by, bz] (dtype=float)
    """
    t = np.asarray(t, dtype=float)
    b_t = np.zeros((t.size, 3), dtype=float)
    for i, tt in enumerate(t):
        rho_ss = get_steady_state(builder, model, float(tt))
        # defensive: ensure rho_ss is a matrix
        rho_ss = np.asarray(rho_ss, dtype=np.complex128)
        b_t[i] = bloch(rho_ss)  # uses existing helper: returns [bx, by, bz]
    return b_t



#------------------------------------
# Rotating frame functions
#------------------------------------
def derivative_vectors_uniform(v_t, dt):
    v_t = np.asarray(v_t, dtype=float)
    vdot = np.empty_like(v_t)

    vdot[1:-1] = (v_t[2:] - v_t[:-2]) / (2 * dt)
    vdot[0] = (-3*v_t[0] + 4*v_t[1] - v_t[2]) / (2 * dt)
    vdot[-1] = (3*v_t[-1] - 4*v_t[-2]) / (2 * dt)

    return vdot


def nearest_grid_index(t, tval):
    idx = np.searchsorted(t, tval)

    if idx <= 0:
        return 0
    if idx >= len(t):
        return len(t) - 1

    if abs(tval - t[idx - 1]) <= abs(t[idx] - tval):
        return idx - 1

    return idx


def su2_z_to_n(n, eps=1e-14):
    """
    Returns U such that U sigma_z U^\dagger = n.sigma.
    Equivalently, U maps the +z Bloch direction to n.
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)

    z = np.array([0.0, 0.0, 1.0])

    c = np.clip(np.dot(z, n), -1.0, 1.0)

    if np.linalg.norm(n - z) < eps:
        return np.eye(2, dtype=complex)

    if np.linalg.norm(n + z) < eps:
        # pi rotation about x maps +z to -z
        return -1j * Pauli.sx

    axis = np.cross(z, n)
    axis = axis / np.linalg.norm(axis)

    theta = np.arccos(c)

    sigma_axis = (
        axis[0] * Pauli.sx
        + axis[1] * Pauli.sy
        + axis[2] * Pauli.sz
    )

    return (
        np.cos(theta / 2) * np.eye(2, dtype=complex)
        - 1j * np.sin(theta / 2) * sigma_axis
    )


def sigma_plus_minus_along_n(n):
    """
    Returns sigma_+^n and sigma_-^n.

    Convention:
        sigma_+ = |+z><-z| = (sx + i sy)/2
        sigma_- = |-z><+z| = (sx - i sy)/2

    sigma_+^n pumps toward +n.
    sigma_-^n pumps toward -n.
    """
    sigma_plus_z = 0.5 * (Pauli.sx + 1j * Pauli.sy)
    sigma_minus_z = 0.5 * (Pauli.sx - 1j * Pauli.sy)

    U = su2_z_to_n(n)

    sigma_plus_n = U @ sigma_plus_z @ U.conj().T
    sigma_minus_n = U @ sigma_minus_z @ U.conj().T

    return sigma_plus_n, sigma_minus_n