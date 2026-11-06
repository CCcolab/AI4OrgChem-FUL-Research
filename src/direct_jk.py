"""Existing exact shell-direct helper; not a source-SCF launcher."""
import numpy as np

def direct_pair(mol,P,threshold,guard):
    from pyscf import gto,scf
    from pyscf.scf import _vhf
    guard()
    with mol.with_integral_screen(threshold):
        cutoff=float(mol._env[gto.mole.PTR_EXPCUTOFF])
        if abs(cutoff-abs(float(np.log(threshold))))>1e-10:
            raise ValueError('Primitive cutoff differs from frozen direct threshold')
        # Official PySCF2.14 initialization; q_cond may evaluate shell integrals.
        # The caller has already charged this J/K pair before optimizer creation.
        opt=_vhf._VHFOpt(mol,'int2e','CVHFnrs8_prescreen',
            'CVHFnr_int2e_q_cond','CVHFnr_dm_cond',threshold)
        opt.direct_scf_tol=threshold
        observed=float(opt.direct_scf_tol)
        if abs(observed/threshold-1)>1e-12:raise ValueError('VHF cutoff not adopted')
        guard()
        J,K=scf.hf.get_jk(mol,np.asarray(P,order='C'),hermi=1,vhfopt=opt,with_j=True,with_k=True)
        info={'primitive_integral_threshold':threshold,'environment_expcutoff':cutoff,
            'VHF_direct_scf_tol':observed,'optimizer_class':'pyscf.scf._vhf._VHFOpt',
            'prescreen':'CVHFnrs8_prescreen','qcond':'CVHFnr_int2e_q_cond',
            'dmcond':'CVHFnr_dm_cond','full_range_Coulomb':True,
            'density_fitting_used':False,'packed_full_ERI_cache_used':False}
    return J,K,info
