import numpy as np
from pyscf import gto, scf

mol = gto.M(atom = """O          0.00000        0.00000        0.11779
  H          0.00000        0.75545       -0.47116
  H          0.00000       -0.75545       -0.47116""", basis = "sto3g", symmetry=True)

mf = scf.RHF(mol).run()

mo_ovlp = np.einsum("ai,bj,ab->ij", mf.mo_coeff, mf.mo_coeff, mf.get_ovlp())
print(np.round(mo_ovlp, decimals=2))

print(np.round(mf.mo_coeff, decimals=2))
idx = np.argmax(abs(mf.mo_coeff.real), axis=0)
print(idx)