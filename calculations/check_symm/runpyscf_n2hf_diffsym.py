from pyscf import gto, scf, symm
import numpy as np

basis = "sto-3g"
verbose = 0
results = []

for i in range(27):
    ilen = round(0.95 + 0.15 * i, 2)
    print(f"====== bond length: {ilen} Å ======")

    atom = f"N 0.0 0.0 0.0; N 0.0 0.0 {ilen}"

    mol_d2h = gto.Mole()
    mol_d2h.build(atom=atom, basis=basis, symmetry=True, verbose=verbose, )
    mf_d2h = scf.RHF(mol_d2h)
    mf_d2h.kernel()

    mol_c1 = gto.Mole()
    mol_c1.build(atom=atom, basis=basis, symmetry=False, verbose=verbose, )
    mf_c1 = scf.RHF(mol_c1)
    mf_c1.kernel()

    for iorb in range(len(mf_d2h.mo_energy)):
    #    print(mf_d2h.mo_energy[iorb] - mf_c1.mo_energy[iorb])
        print(mf_d2h.mo_coeff[iorb] )
        print('===')
        print( mf_c1.mo_coeff[iorb])
        print('================================================')



