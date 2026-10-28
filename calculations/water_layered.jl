# ===========================================================================
# water_layered.jl
#
# 对应 old_BinSim/_try/backup/multi_layer/water_sto3g/water_layer_{abab,aabb}.jl。
# H2O/STO-3G：UCCSD 池重复 1..30 层，比较 abab / aabb 两种排法。
#
# Binsim_10.27 的 run_vqe 直接接受 idxs，所以叠层不再需要 layered_pool：
#   abab: repeat(1:nops, nlayers)
#   aabb: [i for i in 1:nops for _ in 1:nlayers]
# 这里的公共 helper run_pool_vqe 已经封装了这两条规则。
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


function run_single(mole, basis, ham, pool, e_fci; nlayers::Int=1, order::String="abab",
    label::String="UCCSD")
    return run_pool_vqe(basis, ham, pool, e_fci;
        label="$(label) nlayers=$(nlayers) order=$(order)",
        nlayers=nlayers, order=order)
end


function water_layered()
    R_OH = 0.95785
    theta_HOH = deg2rad(104.5)
    sin_ = R_OH * sin(theta_HOH / 2)
    cos_ = R_OH * cos(theta_HOH / 2)

    atom = "
    H 0.0 $(cos_) $(-sin_);
    O 0.0 0.0 0.0;
    H 0.0 $(cos_) $(sin_);
    "

    basis_name = "sto-3g"
    mole = molecule_from_geometry(atom, basis_name;
        frozen_core = 0,
        run_fci = false,
        symmetry = true,
        read_jld2 = false,
        save_name = "water-$(R_OH)-$(basis_name)",
    )

    basis = BasisManager(mole)
    ham   = JW_hamiltonian(mole)

    e_fci, _ = fci_reference(basis, ham)
    println("\nFCI energy = $(e_fci)\n")

    # 全空间 UCCGSD（generalize=true）
    pool_uccgsd = reduce_uccsd_op(mole; ifgeneral=true)
    e_uccgsd = run_single(mole, basis, ham, pool_uccgsd, e_fci;
        nlayers=1, order="abab", label="UCCGSD")

    # 标准 UCCSD，1..30 层，分别用 abab / aabb
    pool_uccsd = reduce_uccsd_op(mole; ifgeneral=false)
    for order in ("abab", "aabb")
        results = Float64[]
        for ilayer in 1:30
            push!(results, run_single(mole, basis, ham, pool_uccsd, e_fci;
                nlayers=ilayer, order=order, label="UCCSD"))
        end
        println("order = $(order), 1..30 layers:")
        println(results)
    end

    println("uccgsd_result = $(e_uccgsd)")
end


if abspath(PROGRAM_FILE) == @__FILE__
    water_layered()
end
