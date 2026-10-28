# ===========================================================================
# n2_frozen_core_scan.jl
#
# 对应 old_BinSim/_try/backup/check_frozencore/frozen4_n2.jl。
# N2/STO-3G、frozen_core=4，扫描键长，比较 FCI 与 Trotterized UCCSD。
#
# 与旧版的差异：
#   - 旧版还跑了 run_exact_vqe（精确 UCCSD），Binsim_10.27 删除了 method.jl，
#     这里只保留 Trotterized UCCSD；需要精确 UCC 时再单独移植 run_exact_vqe；
#   - orbsym 的 `.% 10` 已在 molecule_from_geometry 内部完成，无需重复。
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


function run_uccsd(mole)
    basis = BasisManager(mole)
    ham   = JW_hamiltonian(mole)

    e_fci, _ = fci_reference(basis, ham)
    println("FCI energy = $(e_fci)\n")

    println("=== Running VQE with UCCSD pool (only singles and doubles) ===")
    pool_uccsd = reduce_uccsd_op(mole)
    println("UCCSD pool size: ", length(pool_uccsd))

    println("run Trotterized UCCSD... ...")
    e_trot = run_pool_vqe(basis, ham, pool_uccsd, e_fci; label="Trotterized UCCSD")

    return e_fci, e_trot
end


function scan_n2_frozen4()
    basis_name = "sto-3g"
    results = Tuple{Float64, Float64, Float64}[]

    for ilen in 0:27
        bond_length = round(0.95 + 0.15 * ilen, digits=2)
        println("=============================================")
        println("=== bond length: $(bond_length) Å ===")
        println("=============================================")

        atom = "N 0.0 0.0 0.0; N 0.0 0.0 $(bond_length)"
        mole = molecule_from_geometry(atom, basis_name;
            frozen_core = 4,
            run_fci = false,
            symmetry = true,
            read_jld2 = false,
            save_name = "n2-$(bond_length)-$(basis_name)-fc4",
        )

        e_fci, e_trot = run_uccsd(mole)
        push!(results, (bond_length, e_fci, e_trot))
    end

    println("\n" * repeat("=", 80))
    println("                    Summary: N₂  frozen_core=4  $(basis_name)")
    println(repeat("=", 80))
    @printf("%-12s%-20s%-20s%-20s\n", "Bond (Å)", "FCI", "Trot UCCSD", "Δ(trot - FCI)")
    println(repeat("-", 80))
    for (bond, efci, etrot) in results
        @printf("%-12.2f%-20.14f%-20.14f%.3e\n", bond, efci, etrot, etrot - efci)
    end
    println(repeat("=", 80))
end


if abspath(PROGRAM_FILE) == @__FILE__
    scan_n2_frozen4()
end
