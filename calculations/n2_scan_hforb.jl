# ===========================================================================
# n2_scan_hforb.jl
#
# 对应 old_BinSim/_try/backup/old/_scan_n2.jl。
# 演示用上一条几何的 HF MO 系数作为下一条几何的 RHF 初猜：
#   init_hforb = nothing
#   build_with_init_hforb(mole, init_hforb)
#   init_hforb = mole.hforb
#
# 说明：
#   - mole.hforb 现在是全基组 MO 系数（nao × nao），可以直接喂给 mf.mo_coeff；
#   - 这里比值 ratio 的约定与旧版一致：bond length = ratio * 1.1 Å；
#   - 去掉 run_exact_vqe，只保留 FCI + Trotterized UCCSD。
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
    pool_uccsd = reduce_uccsd_op(mole)
    println("UCCSD pool size: ", length(pool_uccsd))

    e_trot = run_pool_vqe(basis, ham, pool_uccsd, e_fci; label="Trotterized UCCSD")
    return e_fci, e_trot
end


function scan_n2()
    mole = Mole()
    mole.name  = "n2"
    mole.basis = "sto-3g"

    init_hforb = nothing

    for ilen in 0:51
        mole.ratio = (0.8 + 0.2 * ilen) / 1.1
        println("bond length: ", mole.ratio * 1.1)

        build_with_init_hforb(mole, init_hforb)
        init_hforb = mole.hforb

        mole.orbsym = Int64.(mole.orbsym .% 10)
        run_uccsd(mole)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    scan_n2()
end
