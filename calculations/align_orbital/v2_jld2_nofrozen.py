from pyscf import gto, scf
import numpy as np
import h5py, os, opt_einsum
from scipy.linalg import svd

basis = "sto-3g"
last_mo_coeff = None

def align_coeff(C0, C, S):
    M = C0.T @ S @ C
    U, Sigma, Vt = svd(M)
    R = U @ Vt
    C_aligned = C @ R.T
    return C_aligned, R

jld2_dir = os.path.join(os.path.dirname(__file__), "..", "jld2file", basis)
os.makedirs(jld2_dir, exist_ok=True)

for i in range(27):
    ilen = round(0.95 + 0.15 * i, 2)
    print(f"====== bond length: {ilen} Å ======")

    atom = f"N 0.0 0.0 0.0; N 0.0 0.0 {ilen}"

    mol = gto.Mole()
    mol.build(atom=atom, basis=basis, symmetry=False, verbose=3)
    hf = scf.RHF(mol)
    hf.mo_coeff = last_mo_coeff
    hf.kernel()
    S = hf.get_ovlp()

    C_cur = hf.mo_coeff

    if last_mo_coeff is not None:
        C_aligned, R = align_coeff(last_mo_coeff, C_cur, S)
    else:
        C_aligned = C_cur
        R = np.eye(len(C_cur))

    last_mo_coeff = C_aligned

    # Compute MO integrals in original C1 basis, then rotate by R
    hcore = mol.intor("int1e_nuc") + mol.intor("int1e_kin")
    eri   = mol.intor("int2e")

    h1_cur = C_cur.T @ hcore @ C_cur
    h2_cur = opt_einsum.contract("ijkl,ip,jq,kr,ls->pqrs", eri, C_cur, C_cur, C_cur, C_cur)

    h1_aligned = R @ h1_cur @ R.T
    h2_aligned = opt_einsum.contract("pqrs,ip,jq,kr,ls->ijkl", h2_cur, R, R, R, R)

    norb    = mol.nao_nr()
    nelec   = mol.nelec
    enuc    = mol.energy_nuc()
    orbsym  = np.zeros(norb, dtype=np.int64)

    # Save .npy
    np.save(f"n2_{ilen:.2f}.npy", {
        "norb": norb, "nelec": np.array(list(nelec), dtype=np.int64),
        "orbsym": orbsym, "energy_nuc": enuc,
        "one_body_mo": h1_aligned, "two_body_mo": h2_aligned,
        "e_scale": hf.e_tot, "hforb": C_aligned, "R": R,
    }, allow_pickle=True)

    # Save .jld2 (HDF5) — path must match BinSim convention
    jld2_name = f"n2-{ilen:.2f}-{basis}.jld2"
    jld2_path = os.path.join(jld2_dir, jld2_name)
    with h5py.File(jld2_path, "w") as f:
        f["norb"]        = norb
        f["nelec"]       = np.array(list(nelec), dtype=np.int64)
        f["orbsym"]      = orbsym
        f["energy_nuc"]  = enuc
        f["one_body_mo"] = h1_aligned
        f["two_body_mo"] = h2_aligned
        f["e_scale"]     = hf.e_tot
        f["hforb"]       = C_aligned
    print(f"Saved {jld2_path}")

    # Verify orthogonality
    check = C_aligned.T @ S @ C_aligned
    offdiag = np.max(np.abs(check - np.eye(norb)))
    print(f"  orthogonality: max|C^TSC - I| = {offdiag:.2e}")
