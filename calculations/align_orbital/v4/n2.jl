# ===========================================================================
# align_orbital/v4/n2.jl
#
# 对应 old_BinSim/_try/backup/align_orbital/v4/n2.jl。
# 与 v3_yeah/n2.jl 相同，只是键长从 ilen=2 开始（0:1 的缓存缺失时跳过）。
# 已知问题见 v3_yeah/n2.jl 顶部注释。
# ===========================================================================

include(joinpath(@__DIR__, "..", "..", "common.jl"))


function reduce_uccsd_op(mole; Ti::DataType=UInt32, Tv::DataType=Float64,
    complete::Bool=false, ifgeneral::Bool=false)
    orbs = Orbitals()
    kernel(mole, orbs, generalize=ifgeneral)

    g = a -> length(a) in (2, 4)
    orbs.spin_based = orbs.spin_based[g.(orbs.spin_based)]

    return FEB(orbs, Ti=Ti, Tv=Tv, complete=complete)
end


function run_uccsd(mole)
    basis = BasisManager(mole)
    ham   = JW_hamiltonian(mole)

    pool_uccsd = reduce_uccsd_op(mole)
    println("UCCSD pool size: ", length(pool_uccsd))

    return run_pool_vqe(basis, ham, pool_uccsd, 0.0; label="Trotterized UCCSD")
end


function scan_n2()
    basis_name = "sto-3g"
    results = Tuple{Float64, Float64}[]

    for ilen in 2:26
        bond_length = round(0.95 + 0.15 * ilen, digits=2)
        println("====== bond length: $(bond_length) Å ======")

        atom = "N 0.0 0.0 0.0; N 0.0 0.0 $(bond_length)"
        mole = molecule_from_geometry(atom, basis_name;
            frozen_core = 0,
            run_fci = false,
            symmetry = false,
            read_jld2 = true,
            save_name = "n2-fc4-$(bond_length)",
        )

        e_trot = run_uccsd(mole)
        push!(results, (bond_length, e_trot))
    end

    println(rpad("Bond (Å)", 12), rpad("Trot UCCSD", 20))
    for (bond, etrot) in results
        @printf("%-12.2f%-20.14f\n", bond, etrot)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    scan_n2()
end
