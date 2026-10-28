# ===========================================================================
# n2_layered.jl
#
# 对应 old_BinSim/_try/backup/multi_layer/old/n2_layer.jl。
# N2/STO-3G、frozen_core=4，扫描键长，比较 UCCSD 的 1/2/3 层 Trotterized VQE。
# ===========================================================================

include(joinpath(@__DIR__, "common.jl"))


function reduce_uccsd_op(mole; Ti::DataType=UInt32, Tv::DataType=Float64,
    complete::Bool=false, ifgeneral::Bool=false)
    orbs = Orbitals()
    kernel(mole, orbs, generalize=ifgeneral)

    g = a -> length(a) in (2, 4)
    orbs.spin_based = orbs.spin_based[g.(orbs.spin_based)]

    return FEB(orbs, Ti=Ti, Tv=Tv, complete=complete)
end


function run_uccsd(mole, pool, e_fci; nlayers::Int=1)
    basis = BasisManager(mole)
    ham   = JW_hamiltonian(mole)

    println("=== Running VQE with UCCSD pool (nlayers=$(nlayers)) ===")
    println("UCCSD pool size: ", length(pool))
    println("VQE params: ", length(pool) * nlayers)

    e_trot = run_pool_vqe(basis, ham, pool, e_fci;
        label="Trotterized UCCSD nlayers=$(nlayers)", nlayers=nlayers)
    return e_trot
end


function scan_n2_frozen4()
    basis_name = "sto-3g"
    results_1l = Tuple{Float64, Float64}[]
    results_2l = Tuple{Float64, Float64}[]
    results_3l = Tuple{Float64, Float64}[]

    for ilen in 0:26
        bond_length = round(0.95 + 0.15 * ilen, digits=2)
        println("====== bond length: $(bond_length) Å ======")

        atom = "N 0.0 0.0 0.0; N 0.0 0.0 $(bond_length)"
        mole = molecule_from_geometry(atom, basis_name;
            frozen_core = 4,
            run_fci = false,
            symmetry = true,
            read_jld2 = false,
            save_name = "n2-$(bond_length)-$(basis_name)-fc4",
        )

        basis = BasisManager(mole)
        ham   = JW_hamiltonian(mole)
        e_fci, _ = fci_reference(basis, ham)
        pool = reduce_uccsd_op(mole)

        e1 = run_uccsd(mole, pool, e_fci; nlayers=1)
        e2 = run_uccsd(mole, pool, e_fci; nlayers=2)
        e3 = run_uccsd(mole, pool, e_fci; nlayers=3)

        push!(results_1l, (bond_length, e1))
        push!(results_2l, (bond_length, e2))
        push!(results_3l, (bond_length, e3))
    end

    println(repeat("=", 80))
    println(rpad("Bond (Å)", 12), rpad("1-layer", 20), rpad("2-layer", 20), rpad("3-layer", 20))
    println(repeat("-", 80))
    for i in eachindex(results_1l)
        b1, e1 = results_1l[i]
        _,  e2 = results_2l[i]
        _,  e3 = results_3l[i]
        @printf("%-12.2f%-20.14f%-20.14f%-20.14f\n", b1, e1, e2, e3)
    end
    println(repeat("=", 80))
end


if abspath(PROGRAM_FILE) == @__FILE__
    scan_n2_frozen4()
end
