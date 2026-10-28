# ===========================================================================
# align_orbital/v3_yeah/n2.jl
#
# 对应 old_BinSim/_try/backup/align_orbital/v3_yeah/n2.jl。
# 读取同目录 runpyscf_n2.py 生成的“轨道对齐 + frozen_core=4”缓存，
# 再跑 Trotterized UCCSD 扫描。
#
# 注意（已知问题）：
#   runpyscf_n2.py 用 h5py 写 `n2-fc4-<bond>-sto-3g.jld2`，并写到脚本目录下的
#   jld2file/<basis>/ 子目录；JLD2.jl 未必能读 h5py 生成的文件，且路径与
#   Binsim_10.27/jld2file/ 不一致。若 read_jld2 读取失败，本脚本会退回
#   molecule_from_geometry 现场调用 PySCF（此时并没有用上对齐轨道）。
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

    for ilen in 0:26
        bond_length = round(0.95 + 0.15 * ilen, digits=2)
        println("====== bond length: $(bond_length) Å ======")

        atom = "N 0.0 0.0 0.0; N 0.0 0.0 $(bond_length)"
        mole = molecule_from_geometry(atom, basis_name;
            frozen_core = 0,              # 积分已在 runpyscf_n2.py 里冻结
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
