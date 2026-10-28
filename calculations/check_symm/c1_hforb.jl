# ===========================================================================
# check_symm/c1_hforb.jl
#
# 对应 old_BinSim/_try/backup/check_symm/all_c1-hforb/n2.jl。
# 用 C1（symmetry=false）的 hforb/积分，再手动改 orbsym，检查 VQE 结果对
# 轨道对称性标签的敏感度。
#
# 前置条件：先跑过 n2_frozen_core_scan.jl（生成 read_jld2 需要的缓存）。
# 与旧版的差异：去掉 run_exact_vqe，只保留 Trotterized UCCSD。
# ===========================================================================

include(joinpath(@__DIR__, "..", "common.jl"))


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

    e_trot = run_pool_vqe(basis, ham, pool_uccsd, 0.0; label="Trotterized UCCSD")
    return e_trot
end


function scan_n2_frozen4()
    basis_name = "sto-3g"
    results = Tuple{Float64, Float64}[]

    for ilen in 0:26
        bond_length = round(0.95 + 0.15 * ilen, digits=2)
        println("====== bond length: $(bond_length) Å ======")

        atom = "N 0.0 0.0 0.0; N 0.0 0.0 $(bond_length)"
        mole = molecule_from_geometry(atom, basis_name;
            frozen_core = 4,
            run_fci = false,
            symmetry = true,
            read_jld2 = true,
            save_name = "n2-$(bond_length)-$(basis_name)-fc4",
        )

        if ilen in (0, 1)
            mole.orbsym = [6, 7, 0, 2, 3, 5]
        else
            mole.orbsym = [0, 6, 7, 2, 3, 5]
        end

        e_trot = run_uccsd(mole)
        push!(results, (bond_length, e_trot))
    end

    println(rpad("Bond (Å)", 12), rpad("Trot UCCSD", 20))
    for (bond, etrot) in results
        @printf("%-12.2f%-20.14f\n", bond, etrot)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    scan_n2_frozen4()
end
