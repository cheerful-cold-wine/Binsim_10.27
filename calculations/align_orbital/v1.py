from pyscf import gto, scf
import numpy as np
from scipy.linalg import norm, svd

basis = "sto-3g"
last_mo_coeff = None

def align_coeff(C0, C, S):
    M = C0.T @ S @ C
    U, Sigma, Vt = svd(M)
    R = U @ Vt
    C_aligned = C @ R.T
    return C_aligned, R

R_list = []
for i in range(27):
    ilen = round(0.95 + 0.15 * i, 2)
    print(f"====== bond length: {ilen} Å ======")

    atom = f"N 0.0 0.0 0.0; N 0.0 0.0 {ilen}"

    mol= gto.Mole()
    mol.build(atom=atom, basis=basis, symmetry=False, verbose=3,)
    hf = scf.RHF(mol)
    hf.mo_coeff = last_mo_coeff
    hf.kernel()
    S = hf.get_ovlp()

    if last_mo_coeff is not None:
        norb = len(hf.mo_coeff)
        C  = hf.mo_coeff # C.T @ S @ C = Id
        C0 = last_mo_coeff
        C_aligned, R = align_coeff(C0, C, S)
        #print(np.round(C.T @ S @ C0, 2))
        #print(np.round(C_aligned.T @ S @ C0, 2))

        R_list.append(R)


    last_mo_coeff = hf.mo_coeff

