from pyscf import gto, scf, mcscf
import numpy as np
import h5py, os, opt_einsum
from scipy.linalg import svd

basis = "sto-3g"
frozen_core = 4
last_mo_coeff = None
last_active_mo_coeff = None

jld2_dir = os.path.join(os.path.dirname(__file__), "..", "jld2file", basis)
os.makedirs(jld2_dir, exist_ok=True)

for i in range(27):
    ilen = round(0.95 + 0.15 * i, 2)
    print(f"====== bond length: {ilen} A ======")

    atom = f"N 0.0 0.0 0.0; N 0.0 0.0 {ilen}"

    mol = gto.Mole()
    mol.build(atom=atom, basis=basis, symmetry=True, verbose=3)
    hf = scf.RHF(mol)
    hf.mo_coeff = last_mo_coeff
    hf.kernel()

    S = hf.get_ovlp()
    C_cur = hf.mo_coeff
    last_mo_coeff = C_cur

    norb_full = mol.nao_nr()
    nelec_full = mol.nelec
    enuc_full = mol.energy_nuc()
    ncas = norb_full - frozen_core
    nelecas = (nelec_full[0] - frozen_core, nelec_full[1] - frozen_core)

    # 1. CASCI with canonical MOs (C_cur): core orbitals correctly identified
    hf.mo_coeff = C_cur
    casci = mcscf.CASCI(hf, ncas, nelecas, ncore=frozen_core)
    casci.kernel()

    print(casci.mo_coeff.orbsym)
   
# 有两个角度
# 1. vqe优化导致的不平滑
# 2. 算符顺序带来的不平滑
# 应该分别进行排除校验
