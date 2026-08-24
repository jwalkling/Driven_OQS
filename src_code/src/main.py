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
from scipy.optimize import nnls



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



#-----------------------------------------
# Variational AGP calculations
#-----------------------------------------

#Evaluating dissipators on the steady state
def dissipator_actions(
    rho_ss: np.ndarray,
    jump_operators,
) -> np.ndarray:
    """
    Return

        O_j = D[L_j](rho_ss)

    for every jump operator L_j.

    Returns
    -------
    O : ndarray, shape (n_jumps, d, d)
        O[j] = D[L_j](rho_ss).
    """
    rho_ss = np.asarray(rho_ss, dtype=np.complex128)

    if rho_ss.ndim != 2 or rho_ss.shape[0] != rho_ss.shape[1]:
        raise ValueError("rho_ss must be a square matrix.")

    d = rho_ss.shape[0]
    jump_operators = tuple(jump_operators)

    O = np.empty(
        (len(jump_operators), d, d),
        dtype=np.complex128,
    )

    for j, L in enumerate(jump_operators):
        L = np.asarray(L, dtype=np.complex128)

        if L.shape != (d, d):
            raise ValueError(
                f"jump_operators[{j}] has shape {L.shape}; "
                f"expected {(d, d)}."
            )

        Ldag = L.conj().T
        LdagL = Ldag @ L

        O[j] = (
            L @ rho_ss @ Ldag
            - 0.5 * (LdagL @ rho_ss + rho_ss @ LdagL)
        )

    return O

#Finding rotational part of the AGP variationally
def var_AGP_Urot(
    rho_ss: np.ndarray,
    partial_s_rho_ss: np.ndarray,
    hamiltonians,
    lam: float = 0.0,
    *,
    rcond: float = 1e-12,
    return_details: bool = False,
):
    """
    Fit a signed Hamiltonian counterdiabatic term

        H_CD = sum_j x_j H_j

    by minimising

        || partial_s_rho_ss + i sum_j x_j [H_j, rho_ss] ||_F^2
        + lam * ||x||_2^2,

    where x_j are real and may have either sign.

    For lam = 0, this returns the Moore-Penrose minimum-norm
    least-squares solution.

    Parameters
    ----------
    rho_ss : ndarray, shape (d, d)
        Instantaneous steady-state density matrix.

    partial_s_rho_ss : ndarray, shape (d, d)
        Derivative partial_s rho_ss.

    hamiltonians : sequence of ndarray
        Candidate Hermitian operators H_j.

    lam : float, optional
        Non-negative Tikhonov regularisation strength.
        lam = 0 gives the Moore-Penrose solution.

    rcond : float, optional
        Singular-value cutoff used by np.linalg.pinv when lam = 0.

    return_details : bool, optional
        If True, also return diagnostic quantities.

    Returns
    -------
    x : ndarray, shape (n_hamiltonians,)
        Optimal real, signed Hamiltonian coefficients.

    residual : ndarray, shape (d, d)
        Residual at the optimum:

            partial_s_rho_ss + i sum_j x_j [H_j, rho_ss].

    details : dict, optional
        Returned only if return_details=True.
    """
    rho_ss = np.asarray(rho_ss, dtype=np.complex128)
    partial_s_rho_ss = np.asarray(
        partial_s_rho_ss,
        dtype=np.complex128,
    )
    hamiltonians = tuple(
        np.asarray(H, dtype=np.complex128)
        for H in hamiltonians
    )

    if rho_ss.ndim != 2 or rho_ss.shape[0] != rho_ss.shape[1]:
        raise ValueError("rho_ss must be a square matrix.")

    if partial_s_rho_ss.shape != rho_ss.shape:
        raise ValueError(
            "partial_s_rho_ss must have the same shape as rho_ss."
        )

    if len(hamiltonians) == 0:
        raise ValueError("At least one Hamiltonian is required.")

    if lam < 0:
        raise ValueError("lam must be non-negative.")

    if any(H.shape != rho_ss.shape for H in hamiltonians):
        raise ValueError(
            "Every Hamiltonian must have the same shape as rho_ss."
        )

    # C_j = i [H_j, rho_ss].
    #
    # The residual is:
    #
    #     R = partial_s_rho_ss + sum_j x_j C_j.
    commutator_actions = np.array([
        1j * (H @ rho_ss - rho_ss @ H)
        for H in hamiltonians
    ])

    A_complex = np.column_stack([
        C_j.reshape(-1, order="F")
        for C_j in commutator_actions
    ])
    d_complex = partial_s_rho_ss.reshape(-1, order="F")

    # x must be real, so represent the complex Frobenius norm as a
    # real Euclidean norm.
    A_real = np.vstack([A_complex.real, A_complex.imag])
    d_real = np.concatenate([d_complex.real, d_complex.imag])

    # Minimise ||A_real @ x + d_real||^2 + lam ||x||^2.
    if lam == 0.0:
        # Moore-Penrose, minimum-norm least-squares solution.
        x = -np.linalg.pinv(A_real, rcond=rcond) @ d_real
    else:
        # Tikhonov-regularised signed least squares.
        normal_matrix = A_real.T @ A_real + lam * np.eye(len(hamiltonians))
        rhs = -A_real.T @ d_real
        x = np.linalg.solve(normal_matrix, rhs)

    residual = partial_s_rho_ss + np.tensordot(
        x,
        commutator_actions,
        axes=(0, 0),
    )

    if not return_details:
        return x, residual

    residual_norm_squared = float(np.vdot(residual, residual).real)
    x_norm_squared = float(x @ x)

    details = {
        "residual_norm_squared": residual_norm_squared,
        "x_norm_squared": x_norm_squared,
        "objective": residual_norm_squared + lam * x_norm_squared,
        "commutator_actions": commutator_actions,
        "fitted_hamiltonian": sum(
            coefficient * H
            for coefficient, H in zip(x, hamiltonians)
        ),
    }

    return x, residual, details

#Finding dissipative part of the AGP variationally
def var_AGP_Dpop(
    rho_ss: np.ndarray,
    target: np.ndarray,
    jump_operators,
    lam: float = 0.0,
    *,
    return_details: bool = False,
    verbose: bool = False,
):
    """
    Solve the regularised non-negative least-squares problem

        min_{gamma_j >= 0}
            ||sum_j gamma_j D[L_j](rho_ss)
              - target||_F^2
            + lam * ||gamma||_2^2.

    Parameters
    ----------
    rho_ss : ndarray, shape (d, d)
        Instantaneous steady state.

    target : ndarray, shape (d, d)
        This is generally the residual after we minimise the norm of the unitary part, need to mop up what's left with the dissipator.

    jump_operators : sequence of ndarray
        Candidate jump operators L_j.

    lam : float, optional
        Non-negative L2 regularisation parameter.

    return_details : bool, optional
        If False, return only gamma.
        If True, also return diagnostic quantities and Jump objects.

    verbose : bool, optional
        Print the fitted norms. Off by default.

    Returns
    -------
    gamma : ndarray, shape (n_jumps,)
        Optimal non-negative rates.

    details : dict, optional
        Returned only when return_details=True.
    """
    rho_ss = np.asarray(rho_ss, dtype=np.complex128)
    target = np.asarray(
        target,
        dtype=np.complex128,
    )
    jump_operators = tuple(jump_operators)

    if rho_ss.ndim != 2 or rho_ss.shape[0] != rho_ss.shape[1]:
        raise ValueError("rho_ss must be a square matrix.")

    if target.shape != rho_ss.shape:
        raise ValueError(
            "target must have the same shape as rho_ss."
        )

    if lam < 0.0:
        raise ValueError("lam must be non-negative.")

    if len(jump_operators) == 0:
        raise ValueError("At least one jump operator is required.")

    # O_j = D[L_j](rho_ss)
    O = dissipator_actions(rho_ss, jump_operators)

    # Construct the complex design matrix:
    #
    #     A[:, j] = vec(O_j)
    #
    # using the same column-major convention as LiouvillianBuilder.
    A_complex = np.column_stack([
        O_j.reshape(-1, order="F")
        for O_j in O
    ])

    b_complex = target.reshape(-1, order="F")

    # gamma is real, so convert the complex least-squares problem into
    # an equivalent real one.
    A_real = np.vstack([
        A_complex.real,
        A_complex.imag,
    ])
    b_real = np.concatenate([
        b_complex.real,
        b_complex.imag,
    ])

    n_jumps = len(jump_operators)

    # Tikhonov regularisation:
    #
    # ||A gamma - b||^2 + lam ||gamma||^2
    #
    # becomes the augmented NNLS problem
    #
    # || [A; sqrt(lam) I] gamma - [b; 0] ||^2.
    if lam > 0.0:
        A_fit = np.vstack([
            A_real,
            np.sqrt(lam) * np.eye(n_jumps),
        ])
        b_fit = np.concatenate([
            b_real,
            np.zeros(n_jumps),
        ])
    else:
        A_fit = A_real
        b_fit = b_real

    gamma, _ = nnls(A_fit, b_fit)

    if not return_details and not verbose:
        return gamma

    fitted_derivative = np.tensordot(
        gamma,
        O,
        axes=(0, 0),
    )

    residual = fitted_derivative - target

    residual_norm_squared = float(
        np.vdot(residual, residual).real
    )
    gamma_norm_squared = float(gamma @ gamma)
    objective = (
        residual_norm_squared
        + lam * gamma_norm_squared
    )

    if verbose:
        print("Residual norm squared:", residual_norm_squared)
        print("Gamma norm squared:", gamma_norm_squared)
        print("Regularised objective:", objective)

    if not return_details:
        return gamma

    details = {
        "residual_norm_squared": residual_norm_squared,
        "gamma_norm_squared": gamma_norm_squared,
        "objective": objective,
        "fitted_derivative": fitted_derivative,
        "dissipator_actions": O,

        # Directly compatible with builder.build(H, jumps).
        "weighted_jumps": [
            Jump(L, rate)
            for L, rate in zip(jump_operators, gamma)
        ],

        # Keeps the correspondence between rate, operator, and action.
        "terms": [
            {
                "gamma": gamma[j],
                "L": jump_operators[j],
                "O": O[j],
            }
            for j in range(n_jumps)
        ],
    }

    return gamma, details

#Finding the overall AGP variationally
def fit_overall_AGP(
    rho_ss: np.ndarray,
    partial_s_rho_ss: np.ndarray,
    hamiltonians,
    jump_operators,
    *,
    lam_unitary: float = 0.0,
    lam_dissipative: float = 0.0,
    rcond: float = 1e-12,
    return_details: bool = False,
):
    """
    Sequentially fit a Hamiltonian and dissipative AGP.

    Step 1: fit the signed unitary contribution

        partial_s rho_ss + i sum_j x_j [H_j, rho_ss] ~ 0.

    Step 2: let positive dissipative rates reproduce the remaining
    unitary residual:

        sum_k gamma_k D[L_k](rho_ss) ~ residual_unitary,

    where

        residual_unitary
        = partial_s rho_ss + i sum_j x_j [H_j, rho_ss].

    Thus the final residual, in this convention, is

        residual_total
        = residual_unitary - sum_k gamma_k D[L_k](rho_ss).

    Parameters
    ----------
    rho_ss, partial_s_rho_ss : ndarray
        Instantaneous steady state and its derivative.

    hamiltonians : sequence of ndarray
        Candidate Hermitian terms H_j.

    jump_operators : sequence of ndarray
        Candidate jump operators L_k.

    lam_unitary, lam_dissipative : float
        L2 regularisation strengths for x and gamma respectively.

    rcond : float
        Pseudoinverse cutoff for the unitary fit.

    return_details : bool
        If False, return (x, gamma, residual_total).
        If True, additionally return a diagnostics dictionary.

    Returns
    -------
    x : ndarray
        Signed coefficients of the Hamiltonian terms.

    gamma : ndarray
        Non-negative coefficients of the dissipative terms.

    residual_total : ndarray
        Remaining residual after both fits.
    """
    # ---------------------------------------------------------------
    # 1. Unitary contribution:
    #
    # R_U = partial_s rho_ss + i sum_j x_j [H_j, rho_ss].
    # ---------------------------------------------------------------
    x, residual_unitary, unitary_details = var_AGP_Urot(
        rho_ss=rho_ss,
        partial_s_rho_ss=partial_s_rho_ss,
        hamiltonians=hamiltonians,
        lam=lam_unitary,
        rcond=rcond,
        return_details=True,
    )

    # ---------------------------------------------------------------
    # 2. Dissipative contribution:
    #
    # Find D_fit = sum_k gamma_k D[L_k](rho_ss) ~= R_U.
    # ---------------------------------------------------------------
    gamma, dissipative_details = var_AGP_Dpop(
        rho_ss=rho_ss,
        target=residual_unitary,
        jump_operators=jump_operators,
        lam=lam_dissipative,
        return_details=True,
    )

    fitted_dissipative_derivative = (
        dissipative_details["fitted_derivative"]
    )

    # var_AGP_Dpop defines its own residual as D_fit - R_U.
    # Here we use the physically clearer opposite convention:
    #
    # R_total = R_U - D_fit.
    residual_total = (
        residual_unitary - fitted_dissipative_derivative
    )

    if not return_details:
        return x, gamma, residual_total

    total_residual_norm_squared = float(
        np.vdot(residual_total, residual_total).real
    )

    details = {
        "x": x,
        "gamma": gamma,

        "residual_unitary": residual_unitary,
        "residual_unitary_norm_squared": float(
            np.vdot(residual_unitary, residual_unitary).real
        ),

        "fitted_dissipative_derivative":
            fitted_dissipative_derivative,

        "residual_total": residual_total,
        "residual_total_norm_squared":
            total_residual_norm_squared,

        "unitary": unitary_details,
        "dissipative": dissipative_details,

        "H_agp": unitary_details["fitted_hamiltonian"],
        "weighted_jumps": dissipative_details["weighted_jumps"],
    }

    return x, gamma, residual_total, details


#----------------------------------------
# Evolving rho in time w/ approx CD driving
#----------------------------------------
def approx_CD_trajectory(
    model,
    s_of_t,
    times,
    hamiltonians,
    jump_operators,
    *,
    dim=2,
    vec_order="F",
    lam_unitary=0.0,
    lam_dissipative=0.0,
):
    """
    Evolve the instantaneous steady state under the approximate
    counterdiabatic Lindbladian

        L_CD(t) = L(s(t)) + sdot(t) A_s.

    Here `model` is interpreted as a function of s:
        model.H(s)
        model.jumps(s)

    The AGP is calculated variationally on the supplied time grid
    and linearly interpolated between grid points.

    Returns
    -------
    t : ndarray
        Time grid.

    traj : Trajectory
        Trajectory object from LindbladEvolver.

    rho_t : ndarray, shape (nt, d, d)
        CD-evolved density matrices.

    rho_ss_t : ndarray, shape (nt, d, d)
        Instantaneous steady states rho_ss(s(t)).
    """

    # --------------------------------------------------
    # Grid and builder
    # --------------------------------------------------

    t = np.asarray(times, dtype=float)
    s_t = np.array([s_of_t(tt) for tt in t], dtype=float)

    builder = LiouvillianBuilder(
        dim=dim,
        vec_order=vec_order,
    )

    # --------------------------------------------------
    # Instantaneous steady states
    # --------------------------------------------------

    rho_ss_t = np.array([
        get_steady_state(builder, model, s)
        for s in s_t
    ])

    # partial_s rho_ss and sdot
    drho_ds_t = np.gradient(
        rho_ss_t,
        s_t,
        axis=0,
        edge_order=2 if len(t) >= 3 else 1,
    )

    sdot_t = np.gradient(
        s_t,
        t,
        edge_order=2 if len(t) >= 3 else 1,
    )

    # --------------------------------------------------
    # Variational AGP on the grid
    # --------------------------------------------------

    H_agp_t = []
    gamma_agp_t = []

    for rho_ss, drho_ds in zip(rho_ss_t, drho_ds_t):

        _, gamma, _, details = fit_overall_AGP(
            rho_ss=rho_ss,
            partial_s_rho_ss=drho_ds,
            hamiltonians=hamiltonians,
            jump_operators=jump_operators,
            lam_unitary=lam_unitary,
            lam_dissipative=lam_dissipative,
            return_details=True,
        )

        H_agp_t.append(details["H_agp"])
        gamma_agp_t.append(gamma)

    H_agp_t = np.asarray(H_agp_t)
    gamma_agp_t = np.asarray(gamma_agp_t)

    # --------------------------------------------------
    # Linear interpolation
    # --------------------------------------------------

    def interp(tt, values):
        if tt <= t[0]:
            return values[0]
        if tt >= t[-1]:
            return values[-1]

        k = np.searchsorted(t, tt) - 1
        a = (tt - t[k]) / (t[k + 1] - t[k])

        return (1 - a) * values[k] + a * values[k + 1]

    # --------------------------------------------------
    # CD model:
    #
    # H -> H + sdot H_AGP
    # gamma_j -> sdot gamma_j
    # --------------------------------------------------

    def H_CD_of_t(tt):

        s = s_of_t(tt)
        sdot = interp(tt, sdot_t)
        H_agp = interp(tt, H_agp_t)

        return model.H(s) + sdot * H_agp

    def jumps_CD_of_t(tt):

        s = s_of_t(tt)
        sdot = interp(tt, sdot_t)
        gamma = interp(tt, gamma_agp_t)

        # Original physical jumps
        jumps = list(model.jumps(s))

        # Counterdiabatic dissipative terms
        jumps += [
            Jump(L, sdot * g)
            for L, g in zip(jump_operators, gamma)
        ]

        return jumps

    CD_model = LindbladModel2LS(
        H_CD_of_t,
        jumps_CD_of_t,
    )

    # --------------------------------------------------
    # Start in rho_ss(s(t0)) and use existing evolver
    # --------------------------------------------------

    rho0 = rho_ss_t[0]

    evolver = LindbladEvolver(
        CD_model,
        builder,
    )

    traj = evolver.simulate(
        rho0,
        t,
    )

    # Density matrices along the actual CD trajectory
    rho_t = np.array([
        traj.rho(k)
        for k in range(len(traj.t))
    ])

    return t, traj, rho_t, rho_ss_t