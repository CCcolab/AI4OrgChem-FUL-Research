"""Frozen Hermitian capacity reduction; no molecule, Fock build or optimizer.

T/Y are complementary orthonormal coordinates in a whitened local pool.
PASS is conditional on numerical inertia, not an interval certificate or FUL.
"""
import numpy as np


def _h(a): return (a + a.conj().T) * .5
def _norm(a): return float(np.linalg.norm(a, 2)) if a.size else 0.


def numerical_inertia(a, *, atol=1e-12, rtol=1e-10):
    a = np.asarray(a)
    ev = np.linalg.eigvalsh(_h(a))
    band = max(atol, rtol * max(1., float(np.max(abs(ev), initial=0.))))
    return {'negative': int(np.sum(ev < -band)),
            'near_zero': int(np.sum(abs(ev) <= band)),
            'positive': int(np.sum(ev > band)),
            'eigenvalues': ev.tolist(), 'numerical_band': band,
            'strict_interval_certificate': False}


def schur_return_reduction(L, T, Y, b, *, atol=1e-12, rtol=1e-10,
                           frame_atol=1e-10):
    """Return removable T directions to Y using a small Schur complement.

    Keeps the most negative M directions; returns all positive directions and
    at most b-n_minus(Y^H L Y) reliable negative directions nearest zero.
    No claim concerns MOM identity, selfconsistency or another reference.
    """
    L, T, Y = [np.asarray(a) for a in (L, T, Y)]
    if (L.ndim != 2 or L.shape[0] != L.shape[1]
            or T.ndim != 2 or Y.ndim != 2
            or T.shape[0] != len(L) or Y.shape != (len(L), len(L)-T.shape[1])):
        raise ValueError('Complete square L and complementary T/Y shapes required')
    if (isinstance(b, (bool, np.bool_)) or int(b) != b or b < 0
            or atol <= 0 or rtol <= 0 or frame_atol <= 0):
        raise ValueError('Nonnegative integer capacity and positive tolerances required')
    b = int(b)
    if any(not np.all(np.isfinite(a)) for a in (L, T, Y)):
        raise ValueError('Nonfinite input')
    if _norm(L-L.conj().T) > frame_atol * max(1., _norm(L)):
        raise ValueError('L is not reliably Hermitian')
    frame = np.column_stack((T, Y))
    if _norm(frame.conj().T@frame-np.eye(len(L))) > frame_atol:
        raise ValueError('T/Y must be complete and jointly orthonormal')
    L = _h(L)
    A = _h(Y.conj().T@L@Y)
    coupling = Y.conj().T@L@T
    ai = numerical_inertia(A, atol=atol, rtol=rtol)
    li = numerical_inertia(L, atol=atol, rtol=rtol)
    out = {'status': 'UNRESOLVED', 'old_transfer_rank': T.shape[1],
           'pool_dimension': len(L), 'capacity': b,
           'old_Y_inertia': ai, 'L_inertia': li,
           'strict_interval_certificate': False,
           'source_selfconsistency_claimed': False}
    if ai['near_zero'] or li['near_zero']:
        out['reason'] = 'Near-zero old Y or full L; no uncontrolled inverse/classification'
        return out
    if ai['negative'] > b:
        out.update(status='OLD_FRAME_CAPACITY_NOT_PASS',
                   reason='Old retained complement already exceeds capacity')
        return out
    solved = np.linalg.solve(A, coupling) if len(A) else np.zeros_like(coupling)
    M = _h(T.conj().T@L@T-coupling.conj().T@solved)
    ev, z = np.linalg.eigh(M)
    mi = numerical_inertia(M, atol=atol, rtol=rtol)
    out.update(M=M, return_M_inertia=mi,
               solve_residual=_norm(A@solved-coupling))
    if mi['near_zero']:
        out['reason'] = 'Near-zero return Schur; no forced spectral split'
        return out
    if li['negative'] != ai['negative'] + mi['negative']:
        out['reason'] = 'Numerical inertia additivity is unresolved'
        return out
    negative = np.flatnonzero(ev < -mi['numerical_band'])
    positive = np.flatnonzero(ev > mi['numerical_band'])
    slack = b-ai['negative']
    q = min(slack, len(negative))
    # eigh is ascending: last negative entries are nearest zero.
    returned_negative = negative[len(negative)-q:] if q else np.empty(0, dtype=int)
    kept_negative = negative[:len(negative)-q] if q else negative
    returned = np.r_[positive, returned_negative].astype(int)
    Z_return, Z_keep = z[:, returned], z[:, kept_negative]
    T_new = T@Z_keep
    Y_new = np.column_stack((Y, T@Z_return))
    yi = numerical_inertia(Y_new.conj().T@L@Y_new, atol=atol, rtol=rtol)
    if yi['near_zero'] or yi['negative'] > b:
        out.update(reason='Reconstructed complement capacity unresolved/not passed',
                   new_Y_inertia=yi)
        return out
    lower = max(0, li['negative']-b)
    if T_new.shape[1] != lower:
        out['reason'] = 'Constructed rank does not match conditional capacity lower bound'
        return out
    new_frame = np.column_stack((T_new, Y_new))
    frame_error = _norm(new_frame.conj().T@new_frame-np.eye(len(L)))
    if frame_error > frame_atol:
        out['reason'] = 'Returned frame orthogonality unresolved'
        return out
    out.update(status='FROZEN_CAPACITY_REDUCTION_PASS', T_new=T_new, Y_new=Y_new,
               returned_coordinates=Z_return, kept_coordinates=Z_keep,
               new_transfer_rank=T_new.shape[1], new_Y_inertia=yi,
               returned_negative_count=q, unused_capacity_slack=max(0, slack-q),
               capacity_only_minimum_rank=lower,
               frame_orthogonality_error=frame_error,
               minimum_scope='fixed L, numerical inertia and capacity only; not identity or nonlinear FUL')
    return out
