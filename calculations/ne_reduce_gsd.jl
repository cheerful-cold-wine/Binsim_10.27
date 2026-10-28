# ===========================================================================
# ne_reduce_gsd.jl
#
# 对应 old_BinSim/_try/Ne.jl（该文件在 last commit 被删除，可从
# `git show HEAD:_try/Ne.jl` 恢复）。
# Ne/cc-pvdz 的 GSD 约化对照：full / T21+T22 / ncore=1,ntop=1 / ncore=1,ntop=2。
#
# 注意：这是四个算例里最重的一个（full GSD 池很大），默认只跑后三组；
# 需要 full GSD 时把下面 `e_full` 那行取消注释。
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
    atom = "Ne 0.0 0.0 0.0;"
    basis_name = "cc-pvdz"

    mole = molecule_from_geometry(atom, basis_name;
        frozen_core = 0,
        run_fci = false,
        symmetry = true,
        read_jld2 = false,
        save_name = "Ne-$(basis_name)",
    )

    println("norb = $(mole.norb), nelec = $(mole.nelec)")
    println("ne = $(sum(mole.nelec)), nsp = $(2 * mole.norb)")

    basis = BasisManager(mole)
    ham   = JW_hamiltonian(mole)

    e_fci, _ = fci_reference(basis, ham)
    println("\nFCI energy = $(e_fci)\n")

    # pool_full = build_pool(mole)   # full GSD（很重，按需打开）
    pool_t21t22 = build_pool(mole; keep=["T21", "T22"])
    pool_t21_1  = build_pool(mole; keep=["T21", "T22"], ncore=1, ntop=1)
    pool_t21_2  = build_pool(mole; keep=["T21", "T22"], ncore=1, ntop=2)

    # e_full = run_pool_vqe(basis, ham, pool_full, e_fci; label="full GSD")
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
