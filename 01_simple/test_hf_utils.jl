using PyCall
using LinearAlgebra
using TensorOperations
using Printf
using Test
include("hf_utils.jl")

# Setup
pyscf = pyimport("pyscf")

mol = pyscf.gto.M(atom = """O          0.00000        0.00000        0.11779
  H          0.00000        0.75545       -0.47116
  H          0.00000       -0.75545       -0.47116""", basis = "631g",)

mf = pyscf.scf.RHF(mol).run()

s = mol.intor_symmetric("int1e_ovlp")
h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
g = permutedims(mol.intor("int2e", aosym = 1), [4, 3, 2, 1])
nelectron = mol.nelectron
n_ao = convert(UInt8, mol.nao_nr())
e_nuc = mf.energy_nuc()
# End setup

@testset "HF Utils" begin
    @testset "vhf tools" begin
        dm = rand(n_ao, n_ao)
        dm = dm' + dm # Needs to be symmetric

        dm_row_major = permutedims(dm, [2, 1])
        pyscf_j, pyscf_k = pyscf.scf.hf.get_jk(mol, dm_row_major)
        pyscf_vhf = pyscf.scf.hf.get_veff(mol, dm_row_major)

        J = get_coloumb(g, dm)
        K = get_exchange(g, dm)
        @test norm(pyscf_j - J) < 1e-13
        @test norm(pyscf_k - K) < 1e-13

        vhf = get_vhf(g, dm)
        @test norm(pyscf_vhf - vhf) < 1e-13
    end

    @testset "rdm tools" begin
        pyscf_rdm1 = mf.make_rdm1()
        dm1 = make_rdm1(mf.mo_occ, mf.mo_coeff)
        @test norm(pyscf_rdm1 - dm1) < 1e-12

        mo_occ = zeros(n_ao)
        mo_occ[1:nelectron÷2] .= 2
        mo_coeff = rand(n_ao, n_ao)
        rand_error = make_rdm1(mo_occ, mo_coeff) - mf.make_rdm1(mo_coeff, mo_occ)
        @test norm(rand_error) < 1e-12
    end

    @testset "fock utils" begin
        dm = mf.make_rdm1()

        pyscf_fock = mf.get_fock()
        my_fock = get_fock(h, g, dm)
        @test norm(my_fock - pyscf_fock) < 1e-12

        mo_energy, mo_coeff = solve_fock(h, g, dm, s)
        @test mf.mo_energy ≈ mo_energy atol = 1e-6


        # Make sure MOs are orthogonal
        # mo_ovlp = mo_coeff * inv(mo_coeff)
        # mo_ovlp = mf.mo_coeff * inv(mf.mo_coeff)
        mo_ovlp = mo_coeff' * s * mo_coeff
        display(mo_coeff)
        println()
        display(mf.mo_coeff)
        println()
        res = svd(mo_ovlp)
        println(res.S)
        # Fₐₒ = get_fock(h, g, dm)
        # @tensor fock_mo[i, j] := mo_coeff[μ, i] * s[μ, ν] * Fₐₒ[ν, ρ] * s[ρ, σ] * mo_coeff[σ, j]
        # println(diag(fock_mo))
        # println(mo_energy)
        # println(mf.mo_energy)


        mo_occ = zeros(n_ao)
        mo_occ[1:nelectron÷2] .= 2

        dm1 = make_rdm1(mo_occ, mo_coeff)
        @test norm(mf.make_rdm1() - dm1) < 1e-6
        @tensor nelec_test = scalar(dm1[μ, ν] * s[μ, ν])
        println(nelec_test)
        @tensor nelec_test = scalar(mf.make_rdm1()[μ, ν] * s[μ, ν])
        println(nelec_test)
    end

    @testset "energy tools" begin
        my_energy = calc_energy_elec(h, g, mf.make_rdm1())[1] + e_nuc
        @test my_energy ≈ mf.e_tot atol = 1e-6
    end
end

# Clean up
using Glob
rm.(glob("tmp*"))