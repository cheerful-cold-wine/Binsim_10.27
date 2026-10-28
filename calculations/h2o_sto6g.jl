# ===========================================================================
# h2o_sto6g.jl
#
# 对应 old_BinSim/_try/h2o_sto6g.jl（该文件在 last commit 被删除，可从
# `git show HEAD:_try/h2o_sto6g.jl` 恢复）。
# H2O/STO-6G 的 GSD 约化对照：T21+T22、ncore=1,ntop=1、ncore=1,ntop=2。
#
# 与旧版的差异：去掉 spinflip 分支，改用新 run_vqe 接口。
# ===========================================================================

include(joinpath(@__DIR__, "common.jl"))


function build_pool(mole; keep::Vector{String}=["T10", "T11", "T20", "T21", "T22"],
    ncore::Int=0, ntop::Int=0)
    ne  = sum(mole.nelec)
    nsp = 2 * mole.norb

    orbs = Orbitals()
    kernel(mole, orbs, generalize=true)
    reduce_gsd_class(orbs, ne; keep=keep)
    filter_gsd_T21(orbs, ne, nsp; ncore=ncore, ntop=ntop)

    return FEB(orbs)
end


function test()
    R_OH = 0.95785
    theta_HOH = deg2rad(104.5)
    sin_ = R_OH * sin(theta_HOH / 2)
    cos_ = R_OH * cos(theta_HOH / 2)

    atom = "
    H 0.0 $(cos_) $(-sin_);
    O 0.0 0.0 0.0;
    H 0.0 $(cos_) $(sin_);
    "

    basis_name = "sto-6g"
    mole = molecule_from_geometry(atom, basis_name;
        frozen_core = 0,
        run_fci = false,
        symmetry = true,
        read_jld2 = false,
        save_name = "water-$(R_OH)-$(basis_key(basis_name))",
    )

    println("norb = $(mole.norb), nelec = $(mole.nelec)")
    println("ne = $(sum(mole.nelec)), nsp = $(2 * mole.norb)")

    basis = BasisManager(mole)
    ham   = JW_hamiltonian(mole)

    e_fci, _ = fci_reference(basis, ham)
    println("\nFCI energy = $(e_fci)\n")

    pool_t21t22 = build_pool(mole; keep=["T21", "T22"])
    pool_t21_1  = build_pool(mole; keep=["T21", "T22"], ncore=1, ntop=1)
    pool_t21_2  = build_pool(mole; keep=["T21", "T22"], ncore=1, ntop=2)

    e_t21t22 = run_pool_vqe(basis, ham, pool_t21t22, e_fci; label="T21+T22")
    e_t21_1  = run_pool_vqe(basis, ham, pool_t21_1,  e_fci; label="T21+T22 (ncore=1,ntop=1)")
    e_t21_2  = run_pool_vqe(basis, ham, pool_t21_2,  e_fci; label="T21+T22 (ncore=1,ntop=2)")

    println(repeat("=", 80))
    @printf("%-24s %9s %9s %18s %14s\n", "pool", "nops", "nparams", "energy", "err vs FCI")
    @printf("%-24s %9d %9d %18.10f %14.3e\n", "T21+T22",       length(pool_t21t22), length(pool_t21t22), e_t21t22, abs(e_t21t22 - e_fci))
    @printf("%-24s %9d %9d %18.10f %14.3e\n", "T21+T22 (1,1)", length(pool_t21_1),  length(pool_t21_1),  e_t21_1,  abs(e_t21_1 - e_fci))
    @printf("%-24s %9d %9d %18.10f %14.3e\n", "T21+T22 (1,2)", length(pool_t21_2),  length(pool_t21_2),  e_t21_2,  abs(e_t21_2 - e_fci))
    println(repeat("=", 80))
end


if abspath(PROGRAM_FILE) == @__FILE__
    test()
end
