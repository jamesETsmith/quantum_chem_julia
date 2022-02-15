using PyCall
using LinearAlgebra
using TensorOperations
using Printf
using Test
include("hf_utils.jl")

pyscf = pyimport("pyscf")

mol = pyscf.gto.M(atom = """O          0.00000        0.00000        0.11779
  H          0.00000        0.75545       -0.47116
  H          0.00000       -0.75545       -0.47116""", basis = "631g",)

mf = pyscf.scf.RHF(mol).run()

s = mol.intor_symmetric("int1e_ovlp")
h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
g = permutedims(mol.intor("int2e", aosym = 1), [4, 3, 2, 1])
e_nuc = mf.energy_nuc()
nelectron = mol.nelectron
n_ao = convert(UInt8, mol.nao_nr())
println(n_ao)



# println(mf.mo_energy)

# Check Fock solve
# pyscf_mo_energy, pyscf_mo_coeff = mf.eig(mf.get_fock(), mf.get_ovlp())
# println(pyscf_mo_coeff[:, 1]' * pyscf_mo_coeff[:, 1])
# println(dot(pyscf_mo_coeff[:, 1], pyscf_mo_coeff[:, 1]))
# println(pyscf_mo_coeff[1, :]' * pyscf_mo_coeff[10, :])
# println(dot(pyscf_mo_coeff[:, 1], pyscf_mo_coeff[:, 10]))

# exit(0)

#
# SCF Check
#

sym_rand = rand(n_ao, n_ao)
sym_rand = dm' + dm
rand_unitary = qr(sym_rand).Q

# Check rand_unitary
# println(diag(rand_unitary' * rand_unitary))

fake_mo_occ = zeros(n_ao, n_ao)
for i = 1:convert(UInt8, nelectron / 2)
  fake_mo_occ[i, i] = 2
end
starting_dm = rand_unitary' * fake_mo_occ * rand_unitary


function mf_scf(h, g, s, dm0, e_nuc)
  dm = copy(dm0)

  δE = 1
  δdm = 1
  last_energy = calc_energy_elec(h, g, dm0)[1]
  mo_occ = zeros(n_ao)
  for i = 1:convert(UInt8, nelectron / 2)
    mo_occ[i] = 2
  end

  @printf "Initial SCF Energy %16.9f\n" last_energy + e_nuc

  # SCF Cycle
  for i = 1:50

    # Solve Roothan-Hall Equations (Helgaker 10.6.17)
    mo_energy, mo_coeff = solve_fock(h, g, dm, s)
    @tensor mo_ovlp[i,j] := mo_coeff[n,i] * mo_coeff[m,j] * s[n,m]
    res = svd(mo_ovlp)
    println(res.S)
    println("MO Energy Error ", norm(mo_energy - mf.mo_energy))
    println("MO COEFF ERROR ", norm(mo_coeff - mf.mo_coeff))
    exit(0)

    # Convergence Check
    new_dm = make_rdm1(mo_occ, mo_coeff)
    energy_elec, e_1b, e_2b = calc_energy_elec(h, g, new_dm)

    # δs
    δE = energy_elec - last_energy
    δdm = norm(new_dm - dm)

    @printf "SCF Energy %16.9f \t δE = %10.3e \t δdm = %10.3e\n" energy_elec + e_nuc δE δdm

    if abs(δE) < 1e-6
      println("CONVERGED")
      break
    end

    # Update DM
    dm = copy(new_dm)
    last_energy = copy(energy_elec)

    exit(0)
  end
end

# mf_scf(h, g, s, starting_dm, e_nuc)
mf_scf(h, g, s, pyscf_rdm1, e_nuc)
println(mf.e_tot)
# function make_fock_ao(hμν, gμνρσ, Dρσ)

#   fμν = h
# end