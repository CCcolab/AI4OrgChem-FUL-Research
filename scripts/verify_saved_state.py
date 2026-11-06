#!/usr/bin/env python3
"""Verify the five published FUL saved states using only NumPy and stdlib.

This is read-only saved-array arithmetic, not SCF, an integral-error certificate,
or a reconstruction of the LFMO preparation. No quantum backend is imported.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import numpy as np

PRIMARY = {
    '5f147c00c6942a2509d1f944096dc6628d5e23f86c6a1416f143b57259153e80':
        {'AO': 81, 'empty': 33, 'cut': 4, 'identity_policy': 'STO_SOURCE'},
    '0c8e6f53823fbd8a65a5730d8b29049fd5614facc14b05cdcf14197ada7131a6':
        {'AO': 148, 'empty': 100, 'cut': 7, 'identity_policy': 'COMMON_AND_STO'},
    'a8b6105e78f3b7d1d46caeb3758bf6ec745b6768a2837d86a3a1c8f789672694':
        {'AO': 218, 'empty': 170, 'cut': 7, 'identity_policy': 'COMMON_AND_STO'},
    '2a8d02863e26a9ec8246b1437dcee859829eda9de00079b26bec611504e045eb':
        {'AO': 251, 'empty': 203, 'cut': 7, 'identity_policy': 'COMMON_AND_STO'},
    '00594cb18bb15faf8a90085b8ee531911b1118f759f9e73d8dd12fb171a2b357':
        {'AO': 318, 'empty': 270, 'cut': 14, 'identity_policy': 'COMMON_AND_STO'},
}
COUNTS = [3, 1, 3, 17, 5, 17, 2]
LIMITS = {'density_fixedpoint_relative': 1e-8, 'OV_Eh': 1e-7,
          'identity_min': .5, 'array_consistency': 1e-10}


def herm(a): return (a + a.conj().T) * .5
def norm(a): return float(np.linalg.norm(a, 2)) if a.size else 0.
def maxabs(a): return float(np.max(np.abs(a), initial=0.))
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()
def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}
def read_json(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def number(x):
    value = complex(np.asarray(x).item())
    if abs(value.imag) > 1e-10: raise ValueError('Unexpected complex scalar')
    return float(value.real)


def safe_file(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Catalog paths must be portable relative POSIX paths')
    parts = PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts:
        raise ValueError('Catalog path must stay inside the public root')
    path = (root / Path(*parts.parts)).resolve(strict=True)
    try: path.relative_to(root)
    except ValueError as exc: raise ValueError('Catalog symlink leaves the public root') from exc
    if not path.is_file(): raise ValueError('Catalog entry does not identify a file')
    return path


def bound_file(root, row, path_key, hash_key, optional=False):
    if path_key not in row:
        if optional: return None
        raise ValueError('Missing catalog field: ' + path_key)
    expected = row.get(hash_key)
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected.lower()):
        raise ValueError('A full SHA-256 is required for ' + path_key)
    path = safe_file(root, row[path_key])
    if digest(path) != expected.lower(): raise ValueError('SHA-256 mismatch: ' + row[path_key])
    return path


def verify_one(root, row, guard):
    guard()
    expected = row['state_sha256'].lower()
    if expected not in PRIMARY: raise ValueError('Catalog state is outside the five frozen primary states')
    spec = PRIMARY[expected]
    bindings = [('state_path', 'state_sha256', False),
                ('validation_path', 'validation_sha256', False),
                ('audit_arrays_path', 'audit_arrays_sha256', True),
                ('identity_reference_path', 'identity_reference_sha256', True),
                ('construction_path', 'construction_sha256', True)]
    paths = {key: bound_file(root, row, key, hashkey, optional)
             for key, hashkey, optional in bindings}
    before = {key: digest(path) for key, path in paths.items() if path is not None}
    state = arrays(paths['state_path'])
    raw_validation = read_json(paths['validation_path'])
    val = raw_validation.get('validation', raw_validation)
    v = {key[len('validation_'):]: value for key, value in state.items() if key.startswith('validation_')}
    if paths['audit_arrays_path'] is not None: v.update(arrays(paths['audit_arrays_path']))
    refs = arrays(paths['identity_reference_path']) if paths['identity_reference_path'] else {}
    construction = read_json(paths['construction_path']) if paths['construction_path'] else None
    n = spec['AO']; W, S, C, P = [state[k] for k in ('W_full', 'S_AO', 'C_AO', 'P_AO')]
    sizes = state['group_sizes'].astype(int).tolist()
    counts = state['group_occupied_counts'].astype(int).tolist()
    occupied = state['actual_occupied_columns'].astype(int)
    empty = state['actual_unoccupied_columns'].astype(int)
    edges = np.cumsum([0, *sizes])
    if 'F_AO_unshifted' in state:
        F, h, J, K, EN = [state[k] for k in ('F_AO_unshifted', 'h_AO', 'J_AO', 'K_AO', 'E_nuclear')]
    else:
        F, h, J, K, EN = [v['fresh_t_' + k] for k in ('F', 'h', 'J', 'K', 'EN')]
    checks = {}
    checks['frozen_AO_and_electron_ledger'] = bool(
        all(a.shape == (n, n) for a in (W, S, C, P, F, h, J, K)) and
        len(sizes) == 7 and sum(sizes) == n and counts == COUNTS and sizes[-1] == spec['cut'] and
        len(occupied) == 48 and len(empty) == spec['empty'] and
        sorted(np.r_[occupied, empty].tolist()) == list(range(n)))
    if not checks['frozen_AO_and_electron_ledger']: raise ValueError('Invalid saved dimensions/occupation map')
    checks['all_numeric_arrays_finite'] = all(np.isfinite(a).all() for a in (W, S, C, P, F, h, J, K))
    checks['saved_matrices_Hermitian'] = max(norm(a-a.conj().T) for a in (S, P, F, h, J, K)) <= 1e-10
    SC = np.zeros_like(S); FC = np.zeros_like(F)
    for a, b in zip(edges[:-1], edges[1:]):
        wg = W[:, a:b]
        SC[a:b, a:b] = herm(wg.conj().T @ S @ wg)
        FC[a:b, a:b] = herm(wg.conj().T @ F @ wg)
    lc = np.linalg.cholesky(herm(SC))
    def gamma(p):
        left = np.linalg.solve(W, p)
        pw = np.linalg.solve(W.conj(), left.T).T
        return lc.conj().T @ pw @ lc
    def fp(pout, pin):
        return float(np.linalg.norm(gamma(pout-pin), 'fro') /
                     max(1., np.linalg.norm(gamma(pout), 'fro')))
    coordinates = np.linalg.solve(W, C); off = coordinates.copy()
    for a, b in zip(edges[:-1], edges[1:]): off[a:b, a:b] = 0
    checks['saved_MO_columns_remain_in_their_roles'] = norm(off) <= 1e-10
    group_occ = []; P_rebuilt = np.zeros_like(P)
    for g, (a, b, k) in enumerate(zip(edges[:-1], edges[1:], counts)):
        cols = occupied[(occupied >= a) & (occupied < b)]
        if len(cols) != k: raise ValueError('Actual group occupations do not match the ledger')
        O = C[:, cols]; group_occ.append(O); P_rebuilt += 2 * O @ O.conj().T
        if 'C_occ_' + str(g) in state:
            checks['occupied_columns_match_group_' + str(g)] = maxabs(O-state['C_occ_' + str(g)]) <= 1e-12
    density_error = float(np.linalg.norm(P_rebuilt-P, 'fro') / max(1., np.linalg.norm(P, 'fro')))
    checks['actual_occupation_density_reconstruction'] = density_error <= 1e-10
    if 'S_cond' in state: checks['stored_conditional_metric'] = maxabs(SC-state['S_cond']) <= 1e-10
    if 'S_cond' in v: checks['validator_conditional_metric'] = maxabs(SC-v['S_cond']) <= 1e-10
    if 'fresh_F_cond' in v: checks['validator_conditional_Fock'] = maxabs(FC-v['fresh_F_cond']) <= 1e-10
    if 'P_occ' in v: checks['validator_actual_density'] = maxabs(P-v['P_occ']) <= 1e-10
    if 'W' in v: checks['validator_working_basis'] = np.array_equal(W, v['W'])
    fock_error = norm(F-h-J+.5*K)
    checks['saved_F_h_J_K_algebra_identity'] = fock_error <= 1e-10
    energy = {'one_electron': number(np.trace(P@h)), 'coulomb': number(.5*np.trace(P@J)),
              'exchange': number(-.25*np.trace(P@K)), 'nuclear': number(EN)}
    energy['total'] = sum(energy.values())
    energy['half_trace_total'] = number(.5*np.trace(P@(h+F))+EN)
    energy['reported_total'] = float(val['energy']['total'])
    checks['energy_assembly_matches_report'] = max(abs(energy['total']-energy['half_trace_total']),
        abs(energy['total']-energy['reported_total'])) <= 1e-8
    groups = []; P_mapped = np.zeros_like(P); virtual_count = 0; unavailable = []
    common_values = []; sto_values = []; initial_diagnostic = []
    source_layers = [state, refs]
    cross = next((layer[key] for layer in source_layers
                  for key in ('cross_overlap_target_STO', 'cross_overlap_target_source') if key in layer), None)
    for g, (a, b, k, O) in enumerate(zip(edges[:-1], edges[1:], counts, group_occ)):
        guard(); wg = W[:, a:b]; L = np.linalg.cholesky(herm(wg.conj().T@S@wg))
        E = np.linalg.solve(L, wg.conj().T).conj().T
        H = herm(E.conj().T@F@E); U = E.conj().T@S@O
        Qv = np.linalg.qr(U, mode='complete')[0][:, k:]
        ev = np.linalg.eigvalsh(herm(Qv.conj().T@H@Qv)); coupling = Qv.conj().T@H@U
        tau = float(val['groups'][g]['tau'])
        if not np.isfinite(tau) or tau < 1e-7: raise ValueError('Original empty-spectrum band is missing or weakened')
        published = np.asarray(val['groups'][g]['virtual_energies'])
        if ev.shape != published.shape: raise ValueError('Published actual empty spectrum has different dimensions')
        virtual_count += len(ev)
        spectrum_error = maxabs(ev-published)
        eig, u = np.linalg.eigh(H); cg = E@u
        scores = np.sum(abs(cg.conj().T@S@O)**2, axis=1)
        order = np.argsort(-scores, kind='stable'); chosen = order[:k]
        gap = float(scores[order[k-1]]-scores[order[k]]) if 0 < k < len(order) else None
        fresh_occ = cg[:, chosen]; P_mapped += 2*fresh_occ@fresh_occ.conj().T
        fresh_identity = float(np.linalg.svd(O.conj().T@S@fresh_occ, compute_uv=False)[-1])
        info = {'group': g, 'dimension': int(b-a), 'occupied_pairs': k, 'actual_empty_count': len(ev),
                'minimum_actual_empty_Eh': float(ev[0]) if len(ev) else None, 'tau_Eh': tau,
                'all_actual_empty_above_original_band': bool(np.all(ev > tau)),
                'spectrum_report_max_difference_Eh': spectrum_error,
                'OV_spectral_Eh': norm(coupling), 'OV_max_element_Eh': maxabs(coupling),
                'occupied_metric_error': norm(U.conj().T@U-np.eye(k)),
                'complete_complement_error': norm(Qv.conj().T@U),
                'fresh_MOM_identity': fresh_identity, 'fresh_MOM_boundary_gap': gap}
        if spec['identity_policy'] == 'COMMON_AND_STO':
            common = next((layer['common_initial_O_' + str(g)] for layer in source_layers
                           if 'common_initial_O_' + str(g) in layer), None)
            source = next((layer['C_source_occ_' + str(g)] for layer in source_layers
                           if 'C_source_occ_' + str(g) in layer), None)
            if common is not None:
                ci = float(np.linalg.svd(common.conj().T@S@O, compute_uv=False)[-1]); common_values.append(ci)
                info['common_initial_identity'] = ci
            if source is not None and cross is not None:
                si = float(np.linalg.svd(source.conj().T@cross.conj().T@O, compute_uv=False)[-1]); sto_values.append(si)
                info['STO_cross_basis_identity'] = si
        else:
            initial = next((refs[key] for key in ('C_occ_' + str(g), 'O_' + str(g), 'common_initial_O_' + str(g)) if key in refs), None)
            if initial is not None:
                initial_diagnostic.append(float(np.linalg.svd(initial.conj().T@S@O, compute_uv=False)[-1]))
        groups.append(info)
    checks['all_actual_empty_directions_rebuilt'] = virtual_count == spec['empty']
    checks['all_actual_empty_eigenvalues_above_original_bands'] = all(g['all_actual_empty_above_original_band'] for g in groups)
    checks['all_actual_empty_spectra_match_saved_report'] = all(g['spectrum_report_max_difference_Eh'] <= 1e-10 for g in groups)
    checks['all_original_OV_gates'] = all(max(g['OV_spectral_Eh'], g['OV_max_element_Eh']) <= 1e-7 for g in groups)
    checks['group_occupation_and_complement_metrics'] = all(max(g['occupied_metric_error'], g['complete_complement_error']) <= 1e-10 for g in groups)
    checks['fresh_MOM_original_identity_and_tie_rules'] = all(g['fresh_MOM_identity'] >= .5 and (g['fresh_MOM_boundary_gap'] is None or g['fresh_MOM_boundary_gap'] >= 1e-10) for g in groups)
    fresh_map_residual = fp(P_mapped, P)
    checks['fresh_saved_Fock_density_map_fixedpoint'] = fresh_map_residual <= 1e-8
    gamma_results = {'fresh_map_relative': fresh_map_residual}
    PF = v.get('P_F', state.get('P_F'))
    if PF is not None:
        gamma_results['stage_occ_F_relative'] = fp(P, PF)
        checks['original_stage_density_fixedpoint'] = gamma_results['stage_occ_F_relative'] <= 1e-8
        checks['stage_density_residual_matches_report'] = abs(gamma_results['stage_occ_F_relative']-float(val['fixed_point_occ_F']['relative'])) <= 1e-12
    else: unavailable.append('original stage P_F arrays were not supplied')
    if 'P_next' in v:
        gamma_results['stored_next_occ_relative'] = fp(v['P_next'], P)
        checks['original_next_density_fixedpoint'] = gamma_results['stored_next_occ_relative'] <= 1e-8
        checks['next_density_residual_matches_report'] = abs(gamma_results['stored_next_occ_relative']-float(val['fixed_point_next_occ']['relative'])) <= 1e-12
        checks['independent_fresh_map_matches_saved_next_density'] = fp(P_mapped, v['P_next']) <= 1e-8
    else: unavailable.append('original P_next arrays were not supplied')
    identity = {'policy': spec['identity_policy'], 'fresh_MOM_gate_independently_checked': True}
    if spec['identity_policy'] == 'COMMON_AND_STO':
        if len(common_values) == 7:
            checks['original_common_initial_identity0p5'] = min(common_values) >= .5
            identity['minimum_common_initial'] = min(common_values)
        else: unavailable.append('common-initial identity arrays were not supplied')
        if len(sto_values) == 7:
            checks['original_STO_cross_basis_identity0p5'] = min(sto_values) >= .5
            identity['minimum_STO_cross_basis'] = min(sto_values)
        else: unavailable.append('STO cross-basis identity bridge was not supplied')
    else:
        identity['STO_cross_basis'] = 'NOT_APPLICABLE'
        identity['initial_to_final_diagnostic'] = initial_diagnostic or 'NOT_SUPPLIED'
        identity['initial_to_final_is_rejection_gate'] = False
    la = np.linalg.cholesky(herm(S)); physical = herm(la.conj().T@P@la)
    rho = float(np.linalg.norm(physical@physical-2*physical, 'fro')/max(1., np.linalg.norm(2*physical, 'fro')))
    natural = np.linalg.eigvalsh(physical)
    physical_diagnostic = {'separate_from_source_gates': True, 'electron_trace': number(np.trace(S@P)),
        'natural_occupation_maximum': float(natural[-1]),
        'natural_occupation_minimum': float(natural[0]),
        'natural_occupations_above_2_plus_1e_minus_6': int(np.count_nonzero(natural > 2+1e-6)),
        'idempotency_relative': rho, 'natural_occupation_distance': float(np.max(np.minimum(abs(natural), abs(natural-2)))),
        'ordinary_physical_idempotency_gate_pass': rho <= 1e-8}
    checks['96_electron_trace'] = abs(physical_diagnostic['electron_trace']-96) <= 1e-8
    checks['all_bound_files_unchanged'] = all(digest(path) == before[key] for key, path in paths.items() if path is not None)
    failed = [name for name, passed in checks.items() if not passed]
    source_evidence = {'verification_kind': 'saved records, not independently rebuilt construction or SCF history',
        'numeric_pass': val.get('numeric_pass'), 'solver_converged': val.get('solver_converged'),
        'energy_three_confirmations': val.get('energy_three_confirmations'),
        'original_C1': val.get('original_c1'), 'original_C2': val.get('original_c2'),
        'construction_identity_pass': construction.get('identity_pass') if isinstance(construction, dict) else 'NOT_SUPPLIED'}
    return {'id': row['id'], 'basis': row.get('basis'), 'state_sha256': expected,
        'status': 'FAIL' if failed else ('PARTIAL_SAVED_ARRAY_VERIFICATION' if unavailable else 'SAVED_ARRAY_VERIFICATION_PASS'),
        'independent_numeric': {'checks': checks, 'failed_checks': failed, 'density_reconstruction_relative': density_error,
            'F_h_J_K_algebra_error_Eh': fock_error, 'energy_Eh': energy, 'Gamma_residuals': gamma_results,
            'actual_empty_count': virtual_count, 'identity': identity, 'groups': groups},
        'source_evidence': source_evidence, 'unavailable_evidence': unavailable,
        'supplementary_physical_diagnostic': physical_diagnostic,
        'bound_file_sha256': before}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Public checkout root containing states/catalog.json')
    parser.add_argument('--basis', default='all', help='all or an id in states/catalog.json')
    args = parser.parse_args(); start = time.monotonic(); root = args.root.resolve(strict=True)
    def guard():
        if time.monotonic()-start > 55: raise TimeoutError('55 second saved-array verification limit')
    try:
        catalog = read_json(safe_file(root, 'states/catalog.json'))
        rows = catalog['states']
        if catalog.get('schema_version') != 1 or not isinstance(rows, list): raise ValueError('Unsupported catalog schema')
        ids = [row['id'] for row in rows]
        if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids): raise ValueError('Catalog ids must be unique strings')
        if set(row['state_sha256'].lower() for row in rows) != set(PRIMARY) or len(rows) != 5:
            raise ValueError('Catalog must identify exactly the five frozen primary states')
        selected = rows if args.basis == 'all' else [row for row in rows if row['id'] == args.basis]
        if not selected: raise ValueError('Unknown --basis catalog id')
        results = [verify_one(root, row, guard) for row in selected]
        failed = any(row['status'] == 'FAIL' for row in results)
        partial = any(row['status'] == 'PARTIAL_SAVED_ARRAY_VERIFICATION' for row in results)
        report = {'schema_version': 1, 'status': 'FAIL' if failed else ('PARTIAL' if partial else 'PASS'),
            'scope': 'Read-only saved-array arithmetic. No SCF, integral evaluation/certification, LFMO reconstruction or universal-basis proof.',
            'states': results, 'original_limits': LIMITS, 'elapsed_seconds': time.monotonic()-start,
            'python_version': platform.python_version(), 'numpy_version': np.__version__,
            'new_JK': 0, 'new_integrals': 0, 'new_Fock': 0, 'new_SCF': 0}
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if failed else (2 if partial else 0)
    except Exception as exc:
        print(json.dumps({'status': 'ERROR', 'error_type': type(exc).__name__, 'error': str(exc),
            'new_JK': 0, 'new_integrals': 0, 'new_Fock': 0, 'new_SCF': 0}, ensure_ascii=False))
        return 3


if __name__ == '__main__': raise SystemExit(main())
