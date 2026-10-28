# ===========================================================================
# _run_pyscf.jl —— Binsim_10.27 的 PySCF 初始化扩展
#
# 从 old_BinSim 的 jl/_run_pyscf.jl 移植而来，保持 "_" 前缀文件的约定：
# 不修改 jl/geo.jl 的 scf_mole / build，只在这里提供额外的初始化入口。
#
# 与旧版的差异：
#   1. Mole 已带 hforb 字段（见 jl/sysinfo.jl），因此这里直接返回 hforb；
#   2. hforb 始终保存“全基组”的 mf.mo_coeff（nao × nao），即使 frozen_core>0
#      也不裁列，这样下一条几何可以直接把它作为 RHF 初猜；
#   3. orbsym 做 `.% 10`，与 geo.jl 的 scf_mole 保持一致；
#   4. 去掉了旧版调试打印。
# ===========================================================================

# 旧 geo.jl 的 molecule_geometry：h2/h4/.../h120 走 h_chain，其余走 mole_geo。
function molecule_geometry(name::String, ratio::Float64)
    if name in ["h$i" for i in 2:2:120]
        nH = parse(Int, match(r"\d+", name).match)
        return h_chain(nH, ratio)
    else
        return mole_geo(name, ratio)
    end
end


function init_scf_with_init_hforb(pyfun, name::String, ratio::Float64, basis::Union{String, Dict{String,String}}, save_path::String, init_hforb::Union{Nothing, Array}=nothing;
    frozen_core::Int64=0,
    run_cisd::Bool=false,
    run_mp2::Bool=false,
    run_ccsd::Bool=false,
    run_fci::Bool=false,
)
    gto = pyimport("pyscf.gto")
    scf = pyimport("pyscf.scf")
    ci  = pyimport("pyscf.ci")
    cc  = pyimport("pyscf.cc")
    fci = pyimport("pyscf.fci")
    mp  = pyimport("pyscf.mp")
    mcscf = pyimport("pyscf.mcscf")

    e_scf   = 0.0
    e_cisd  = 0.0
    e_mp2   = 0.0
    e_ccsd  = 0.0
    e_fci   = 0.0

    geo = molecule_geometry(name, ratio)

    mol = gto.M(atom=geo, basis=basis, spin=0.0, symmetry=true)
    println("Use symmetry. Molecule point group: $(mol.groupname)")

    norb = mol.nao_nr()
    nelec::Tuple{Int64,Int64} = mol.nelec
    energy_nuc::Float64 = mol.energy_nuc()
    println("Norb: $(norb)   Ne: $(nelec)")

    norb > 120 && error("Number of orbitals exceeds 120: $(norb)")

    mf = scf.RHF(mol)
    println("Running RHF...")
    if !isnothing(init_hforb)
        mf.mo_coeff = init_hforb
    end
    mf.kernel()
    e_scf  = mf.e_tot
    e_scale = mf.e_tot
    hforb  = copy(Matrix{Float64}(mf.mo_coeff))   # 全基组 MO 系数

    orbsym = hasproperty(mf, :orbsym) ? mf.orbsym : ones(Int64, norb)
    orbsym = Int64.(orbsym .% 10)

    if run_cisd
        mf_ci = ci.CISD(mf)
        println("Running CISD ...")
        mf_ci.kernel()
        e_cisd = mf_ci.e_tot
        e_scale = mf_ci.e_tot
    end

    if run_mp2
        mf_mp2 = mp.MP2(mf)
        println("Running MP2 ...")
        mf_mp2.kernel()
        e_mp2 = mf_mp2.e_tot
        e_scale = mf_mp2.e_tot
    end

    if run_ccsd
        mf_cc = cc.CCSD(mf)
        println("Running CCSD ...")
        mf_cc.kernel()
        e_ccsd = mf_cc.e_tot
        e_scale = mf_cc.e_tot
    end

    if run_fci
        mf_fci = fci.FCI(mf)
        println("Running FCI ...")
        mf_fci.kernel()
        e_fci = mf_fci.e_tot
        e_scale = mf_fci.e_tot
        println(@sprintf("E(FCI) = %.14f  E_corr = %.14f", e_scale, e_scale - mf.e_tot))
    end

    one_body_mo::Array{Float64,2}, two_body_mo::Array{Float64,4} = pyfun.mol_int(mf)

    if frozen_core > 0
        ncore = frozen_core
        ncas = norb - ncore
        nelecas = (Int64(nelec[1] - ncore), Int64(nelec[2] - ncore))

        println("Freezing $ncore core orbitals. Active space: $ncas orbitals, $nelecas electrons")

        casci = mcscf.CASCI(mf, ncas, nelecas, ncore=ncore)
        one_body_mo, energy_nuc = casci.get_h1eff()
        two_body_mo = two_body_mo[ncore+1:end, ncore+1:end, ncore+1:end, ncore+1:end]
        norb = ncas
        nelec = nelecas
        orbsym = orbsym[ncore+1:end]
        e_scale = energy_nuc
        # 注意：hforb 保留全基组，不在这里裁列。
    end

    dir = dirname(save_path)
    mkpath(dir)
    jldopen(save_path, "w") do file
        file["norb"] = norb
        file["nelec"] = nelec
        file["orbsym"] = orbsym
        file["energy_nuc"] = energy_nuc
        file["one_body_mo"] = one_body_mo
        file["two_body_mo"] = two_body_mo
        file["e_scale"] = e_scale
        file["hforb"] = hforb
    end

    println("Saved to $(save_path)\n")

    return norb, nelec, orbsym, energy_nuc, one_body_mo, two_body_mo, e_scale, hforb
end


function init_scf_from_atom(pyfun, atom::String, basis::Union{String, Dict{String,String}}, save_path::String;
    spin::Float64=0.0,
    charge::Int64=0,
    symmetry::Bool=true,
    init_hforb::Union{Nothing, Array}=nothing,
    frozen_core::Int64=0,
    run_cisd::Bool=false,
    run_mp2::Bool=false,
    run_ccsd::Bool=false,
    run_fci::Bool=false,
)
    gto = pyimport("pyscf.gto")
    scf = pyimport("pyscf.scf")
    ci  = pyimport("pyscf.ci")
    cc  = pyimport("pyscf.cc")
    fci = pyimport("pyscf.fci")
    mp  = pyimport("pyscf.mp")
    mcscf = pyimport("pyscf.mcscf")

    e_scf   = 0.0
    e_cisd  = 0.0
    e_mp2   = 0.0
    e_ccsd  = 0.0
    e_fci   = 0.0

    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, symmetry=symmetry)
    if symmetry
        println("Use symmetry. Molecule point group: $(mol.groupname)")
    end

    norb = mol.nao_nr()
    nelec::Tuple{Int64,Int64} = mol.nelec
    energy_nuc::Float64 = mol.energy_nuc()
    println("Norb: $(norb)   Ne: $(nelec)")

    norb > 120 && error("Number of orbitals exceeds 120: $(norb)")

    mf = scf.RHF(mol)
    println("Running RHF...")
    if !isnothing(init_hforb)
        mf.mo_coeff = init_hforb
    end
    mf.kernel()
    e_scf  = mf.e_tot
    e_scale = mf.e_tot
    hforb  = copy(Matrix{Float64}(mf.mo_coeff))   # 全基组 MO 系数

    orbsym = hasproperty(mf, :orbsym) ? mf.orbsym : ones(Int64, norb)
    orbsym = Int64.(orbsym .% 10)

    if run_cisd
        mf_ci = ci.CISD(mf)
        println("Running CISD ...")
        mf_ci.kernel()
        e_cisd = mf_ci.e_tot
        e_scale = mf_ci.e_tot
    end

    if run_mp2
        mf_mp2 = mp.MP2(mf)
        println("Running MP2 ...")
        mf_mp2.kernel()
        e_mp2 = mf_mp2.e_tot
        e_scale = mf_mp2.e_tot
    end

    if run_ccsd
        mf_cc = cc.CCSD(mf)
        println("Running CCSD ...")
        mf_cc.kernel()
        e_ccsd = mf_cc.e_tot
        e_scale = mf_cc.e_tot
    end

    if run_fci
        mf_fci = fci.FCI(mf)
        println("Running FCI ...")
        mf_fci.kernel()
        e_fci = mf_fci.e_tot
        e_scale = mf_fci.e_tot
        println(@sprintf("E(FCI) = %.14f  E_corr = %.14f", e_scale, e_scale - mf.e_tot))
    end

    one_body_mo::Array{Float64,2}, two_body_mo::Array{Float64,4} = pyfun.mol_int(mf)

    if frozen_core > 0
        ncore = frozen_core
        ncas = norb - ncore
        nelecas = (Int64(nelec[1] - ncore), Int64(nelec[2] - ncore))

        println("Freezing $ncore core orbitals. Active space: $ncas orbitals, $nelecas electrons")

        casci = mcscf.CASCI(mf, ncas, nelecas, ncore=ncore)
        one_body_mo, energy_nuc = casci.get_h1eff()
        two_body_mo = two_body_mo[ncore+1:end, ncore+1:end, ncore+1:end, ncore+1:end]
        norb = ncas
        nelec = nelecas
        orbsym = orbsym[ncore+1:end]
        e_scale = energy_nuc
        # 注意：hforb 保留全基组，不在这里裁列。
    end

    dir = dirname(save_path)
    mkpath(dir)
    jldopen(save_path, "w") do file
        file["norb"] = norb
        file["nelec"] = nelec
        file["orbsym"] = orbsym
        file["energy_nuc"] = energy_nuc
        file["one_body_mo"] = one_body_mo
        file["two_body_mo"] = two_body_mo
        file["e_scale"] = e_scale
        file["hforb"] = hforb
    end

    println("Saved to $(save_path)\n")

    return norb, nelec, orbsym, energy_nuc, one_body_mo, two_body_mo, e_scale, hforb
end
