using LinearAlgebra
using TensorOperations


# 
# vhf tools
#
function get_coloumb(g, D)
    @tensor J[μ, ν] := g[μ, ν, ρ, σ] * D[ρ, σ]
    return J
end

function get_exchange(g, D)
    @tensor K[μ, ν] := g[μ, σ, ρ, ν] * D[ρ, σ]
    return K
end

function get_vhf(g, D)
    return get_coloumb(g, D) - 0.5 * get_exchange(g, D)
end

#
# RDM tools
#
function make_rdm1(mo_occ::Vector{Float64}, mo_coeff::Array{Float64})
    occ_coeff = mo_coeff[:, mo_occ.>0]
    return occ_coeff * diagm(0 => mo_occ[mo_occ.>0]) * occ_coeff'
end

#
# Fock tools
#
function get_fock(h, g, D)
    # This has to be Hermitian or eigenvalue solver with get messed up
    return Hermitian(h + get_vhf(g, D))
end

function solve_fock(h, g, D, s)
    res = eigen(get_fock(h, g, D), s)
    mo_energy = res.values
    mo_coeff = res.vectors
    return mo_energy, mo_coeff
end

function calc_energy_elec(h, g, dm)
    @tensor one_body = scalar(h[μ, ν] * dm[ν, μ])
    vhf = get_vhf(g, dm)
    @tensor two_body = 0.5 * scalar(vhf[μ, ν] * dm[ν, μ])
    return one_body + two_body, one_body, two_body
end

