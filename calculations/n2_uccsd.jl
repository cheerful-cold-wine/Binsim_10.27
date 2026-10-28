# ===========================================================================
# n2_uccsd.jl
#
# 对应 old_BinSim/_try/backup/old/deepseek_uccsd.jl。
# N2/STO-3G 的 UCCSD（仅 singles + doubles）基准：FCI vs Trotterized UCCSD。
#
# 与旧版的差异：去掉 run_exact_vqe；run_vqe 换成新接口。
# 旧文件里的 0:21 扫描默认是注释掉的，这里保持同样的行为，
# 需要完整扫描时把 `for ilen in 0:21` 那段取消注释即可。
# ===========================================================================

include(joinpath(@__DIR__, "common.jl"))


function reduce_uccsd_op(mole; Ti::DataType=UInt32, Tv::DataType=Float64,
    complete::Bool=false)
    orbs = Orbitals()
    kernel(mole, orbs, generalize=false)   # 标准激发：占据 → 虚拟

    # 长度 2 = 单激发，长度 4 = 双激发
    g = a -> length(a) in (2, 4)
    orbs.spin_based = orbs.spin_based[g.(orbs.spin_based)]

    return FEB(orbs, Ti=Ti, Tv=Tv, complete=complete)
end


function run_uccsd(mole)
    basis = BasisManager(mole)
    ham   = JW_hamiltonian(mole)

    println("Computing FCI reference energy ...")
    e_fci, _ = fci_reference(basis, ham)
    println("FCI energy = ", e_fci, "\n")

    println("=== Running VQE with UCCSD pool (only singles and doubles) ===")
    pool_uccsd = reduce_uccsd_op(mole)
    println("UCCSD pool size: ", length(pool_uccsd))

    e_trot = run_pool_vqe(basis, ham, pool_uccsd, e_fci; label="Trotterized UCCSD")
    println("")

    return e_fci, e_trot
end


if abspath(PROGRAM_FILE) == @__FILE__
    mole = Mole()
    mole.name  = "n2"
    mole.basis = "sto-3g"
    mole.ratio = 1.0          # bond length = 1.1 Å

    build(mole)               # 走 jl/geo.jl 的 name/ratio 缓存接口
    run_uccsd(mole)

    # 旧文件的完整扫描（取消注释即可，0:21 共 22 个键长）：
    # for ilen in 0:21
    #     mole.ratio = (0.8 + 0.2 * ilen) / 1.1
    #     println("bond length: ", mole.ratio * 1.1)
    #     build(mole)
    #     run_uccsd(mole)
    # end
end
