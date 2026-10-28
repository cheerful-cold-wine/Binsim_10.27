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

for i in range(2, 27):
    ilen = round(0.95 + 0.15 * i, 2)
    print(f"====== bond length: {ilen} A ======")

    atom = f"N 0.0 0.0 0.0; N 0.0 0.0 {ilen}"

    mol = gto.Mole()
    mol.build(atom=atom, basis=basis, symmetry=False, verbose=3)
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
    h1_active, ecore = casci.get_h1eff()

    # full MO integrals for active-space two-body
    eri   = mol.intor("int2e")
    h2_cur = opt_einsum.contract("ijkl, ip, js, kq, lr -> pqrs", eri, C_cur, C_cur, C_cur, C_cur)
    h2_active = h2_cur[frozen_core:, frozen_core:, frozen_core:, frozen_core:]

    # 2. Align active space only
    C_active = C_cur[:, frozen_core:]

    if last_active_mo_coeff is not None:
        def align_coeff(C0, C, S):# C0 <~ C R^T
            M = C0.T @ S @ C
            U, Sigma, Vt = svd(M)
            R = U @ Vt
            return R
        R_active = align_coeff(last_active_mo_coeff, C_active, S)
    else:
        R_active = np.eye(ncas)

        theta1 = 0.3
        R_active[1, 1] =  np.cos(theta1)
        R_active[1, 2] =  np.sin(theta1)
        R_active[2, 1] = -np.sin(theta1)
        R_active[2, 2] =  np.cos(theta1)
        theta2 = 0.8
        R_active[3, 3] =  np.cos(theta2)
        R_active[3, 4] =  np.sin(theta2)
        R_active[4, 3] = -np.sin(theta2)
        R_active[4, 4] =  np.cos(theta2)

        if False:
            R_active = np.matrix([[1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 1],
                                  [0, 1, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0]]) 


    C_active_aligned = C_active @ R_active.T
    last_active_mo_coeff = C_active_aligned
    
    ####### for check ######

    print(np.round(R_active, 2))    
    #print(np.round(C_active, 2))
    if False:
        print('cicada')
        hf.mo_coeff = np.hstack((C_cur[:, :frozen_core], C_active_aligned))
        casci_check = mcscf.CASCI(hf, ncas, nelecas, ncore=frozen_core)
        casci_check.kernel()
    #######################

    # 3. Rotate active-space integrals by R_active 
    h1_active_aligned = R_active @ h1_active @ R_active.T
    h2_active_aligned = opt_einsum.contract("pqrs,ip,jq,kr,ls->ijkl", h2_active,
                                            R_active, R_active, R_active, R_active)

    # 4. Save .jld2
    if True:
        jld2_name = f"n2-fc4-{ilen}-{basis}.jld2"
        jld2_path = os.path.join(jld2_dir, jld2_name)
        with h5py.File(jld2_path, "w") as f:
            f["norb"]        = ncas
            f["nelec"]       = np.array(list(nelecas), dtype=np.int64)
            f["orbsym"]      = np.zeros(ncas, dtype=np.int64)
            f["energy_nuc"]  = ecore
            f["one_body_mo"] = h1_active_aligned
            f["two_body_mo"] = h2_active_aligned
            f["e_scale"]     = 0.
            f["hforb"]       = C_active_aligned
        print(f"Saved {jld2_path}")

    # Save .npy
    if False:
        np.save(f"n2_{ilen}.npy", {
            "norb_full": norb_full, "nelec_full": np.array(list(nelec_full), dtype=np.int64),
            "norb_active": ncas, "nelec_active": np.array(list(nelecas), dtype=np.int64),
            "energy_nuc": ecore, "energy_nuc_raw": enuc_full,
            "one_body_mo": h1_active_aligned, 
            "two_body_mo": h2_active_aligned,
            "e_scale": 0., 
            "hforb": C_active_aligned,
            "frozen_core": frozen_core, "R_active": R_active,
        }, allow_pickle=True)

    # Verify
    print(f"  CASCI E = {casci.e_tot:.10f}  ecore = {ecore:.10f}  " +
          f"active: norb={ncas} nelec={nelecas}")
