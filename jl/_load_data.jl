# ===========================================================================
# _load_data.jl —— Binsim_10.27 的分子构建扩展
#
# 从 old_BinSim 的 jl/_load_data.jl 移植而来，保持 "_" 前缀文件的约定：
# 不改动 jl/geo.jl 的 build(mole)；sysinfo.jl 只给 Mole 增加了 hforb 字段。
# 本文件提供：
#   - basis_key / parse_basis    : 逐元素基组的缓存名
#   - molecule_from_geometry      : 直接给原子坐标建分子（支持逐元素基组、
#                                   冻结芯、CISD/MP2/CCSD/FCI 参考值）
#   - build_with_init_hforb       : name/ratio 风格的扫描，可复用上一条几何的
#                                   MO 作为 SCF 初猜
#
# Mole 的 hforb 字段在 jl/sysinfo.jl 中定义，语义见该文件注释。
# ===========================================================================

function basis_key(basis::Union{String, Dict{String,String}})
    if basis isa String
        return basis
    else
        return join([string(k, "@", v) for (k, v) in sort(collect(basis))], "+")
    end
end


function parse_basis(s::AbstractString)
    if occursin(';', s) || occursin('=', s)
        d = Dict{String,String}()
        for part in split(s, ';')
            part = strip(part)
            isempty(part) && continue
            kv = split(part, '=', limit=2)
            length(kv) == 2 || error("Invalid basis spec: $(part)")
            d[strip(kv[1])] = strip(kv[2])
        end
        return d
    end
    return s
end


# name/ratio 风格：从 jld2 缓存读，读不到就调用 PySCF。
# init_hforb 非空时作为本次 RHF 的初猜（Mole.hforb 保存的是全基组 MO）。
# 注意：Mole.basis 是 String，所以这条路径只支持单一基组；逐元素基组请用
# molecule_from_geometry。
function build_with_init_hforb(mole::Mole, init_hforb::Union{Nothing, Array}=nothing, filepath::String="")
    if isempty(filepath)
        name  = mole.name
        ratio = mole.ratio
        basis = mole.basis

        filename = "$(name)-$(ratio)-$(basis_key(basis)).jld2"
        filepath = joinpath(jld2path, filename)

        loaded = false
        if isfile(filepath)
            try
                jldopen(filepath, "r") do file
                    mole.norb        = file["norb"]
                    mole.nelec       = file["nelec"]
                    mole.orbsym      = file["orbsym"]
                    mole.energy_nuc  = file["energy_nuc"]
                    mole.one_body_mo = file["one_body_mo"]
                    mole.two_body_mo = file["two_body_mo"]
                    mole.e_scale     = file["e_scale"]
                    mole.hforb       = get(file, "hforb", nothing)
                end
                loaded = true
            catch
            end
        end

        if loaded
            if is_rank0_or_serial()
                println("Successfully read data from: $(abspath(filepath))")
                println("  name: $(name)")
                println("  ratio: $(ratio)")
                println("  basis: $(basis_key(basis))")
            end
        else
            pushfirst!(pyimport("sys")."path", pypath)
            pyfun = pyimport("mole_pbc_int")

            mole.norb, mole.nelec, mole.orbsym, mole.energy_nuc,
            mole.one_body_mo, mole.two_body_mo, mole.e_scale, mole.hforb = init_scf_with_init_hforb(
                pyfun, mole.name, mole.ratio, mole.basis, filepath, init_hforb, run_fci=false)
        end
    else
        filepath = joinpath(jld2path, filepath)
        try
            isfile(filepath) || error("missing cache file: $(filepath)")
            jldopen(filepath, "r") do file
                mole.norb        = file["norb"]
                mole.nelec       = file["nelec"]
                mole.orbsym      = file["orbsym"]
                mole.energy_nuc  = file["energy_nuc"]
                mole.one_body_mo = file["one_body_mo"]
                mole.two_body_mo = file["two_body_mo"]
                mole.e_scale     = file["e_scale"]
                mole.hforb       = get(file, "hforb", nothing)
            end

            if is_rank0_or_serial()
                println("Successfully read data from: $(abspath(filepath))")
            end
        catch
            error("You need to run your algorithm first and save the system information into $(jld2path)")
        end
    end

    norb   = mole.norb
    na, nb = mole.nelec
    ne     = na + nb
    nq     = norb * 2

    if is_rank0_or_serial()
        @printf("  nα: %d, nβ: %d, ne: %d, norb: %d, nq: %d\n\n",
        na, nb, ne, norb, nq)
    end
end


# 直接给原子坐标建分子。basis 可为 String 或 Dict{String,String}（逐元素基组）。
# 返回的 Mole.hforb 为全基组 MO 系数（nao × nao）。
function molecule_from_geometry(atom::String, basis::Union{String, Dict{String,String}};
    spin::Float64=0.0,
    charge::Int64=0,
    symmetry::Bool=true,
    init_hforb::Union{Nothing, Array}=nothing,
    frozen_core::Int64=0,
    read_jld2::Bool=false,
    save_name::String="",
    run_cisd::Bool=false,
    run_mp2::Bool=false,
    run_ccsd::Bool=false,
    run_fci::Bool=false,
)
    pushfirst!(pyimport("sys")."path", pypath)
    pyfun = pyimport("mole_pbc_int")

    if isempty(save_name)
        save_name = string(hash(atom * basis_key(basis)), base=62)
    end

    filename = "$(save_name)-$(basis_key(basis)).jld2"
    save_path = joinpath(jld2path, filename)

    loaded = false
    if read_jld2 && isfile(save_path)
        try
            file = jldopen(save_path, "r")
            norb_        = file["norb"]
            nelec_       = (Int(file["nelec"][1]), Int(file["nelec"][2]))
            orbsym_      = file["orbsym"]
            energy_nuc_  = file["energy_nuc"]
            one_body_mo_ = file["one_body_mo"]
            two_body_mo_ = file["two_body_mo"]
            e_scale_     = file["e_scale"]
            hforb_       = get(file, "hforb", nothing)
            close(file)
            loaded = true
            if is_rank0_or_serial()
                println("Successfully read data from: $(abspath(save_path))")
            end
        catch
        end
    end

    if !loaded
        norb_, nelec_, orbsym_, energy_nuc_,
        one_body_mo_, two_body_mo_, e_scale_, hforb_ = init_scf_from_atom(
            pyfun, atom, basis, save_path;
            spin=spin, charge=charge, symmetry=symmetry,
            init_hforb=init_hforb, frozen_core=frozen_core,
            run_cisd=run_cisd, run_mp2=run_mp2,
            run_ccsd=run_ccsd, run_fci=run_fci,
        )
    end

    na, nb = nelec_
    if is_rank0_or_serial()
        @printf("  nα: %d, nβ: %d, ne: %d, norb: %d, nq: %d\n\n",
        na, nb, na + nb, norb_, norb_ * 2)
    end

    mole = Mole()
    mole.name        = ""                 # 坐标接口没有 name/ratio
    mole.ratio       = 1.0
    mole.basis       = basis_key(basis)   # 只存缓存键，原始 Dict 不保留
    mole.norb        = norb_
    mole.nelec       = nelec_
    mole.orbsym      = orbsym_
    mole.energy_nuc  = energy_nuc_
    mole.one_body_mo = one_body_mo_
    mole.two_body_mo = two_body_mo_
    mole.e_scale     = e_scale_
    mole.hforb       = hforb_

    return mole
end
