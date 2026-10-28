# ===========================================================================
# _ansatz.jl —— Binsim_10.27 的算符池扩展
#
# 从 old_BinSim 的 jl/_ansatz.jl 移植“非自旋匹配”部分：
#   1. unique_operator_pool : 符号规范化去重（X 与 -X 合并）
#   2. reduce_gsd_class     : 按 T10/T11/T20/T21/T22 分类保留 GSD 算符
#   3. filter_gsd_T21       : 在 T21 类中剔除涉及 core / 最高虚轨道的项
#   4. layered_idxs         : 叠层 ansatz 的算子下标序列（abab / aabb）
#
# 本文件在 ansatz.jl 之后 include，因此可以覆盖同名方法，而不需要修改
# 原始 jl/ansatz.jl。
#
# 未移植（按需求排除）：kernel(spin_adapted=true)、generate_ci_spatial_orbitals、
# SpinAdaptedFEB/QEB、自旋翻转 canonicalization 与 FEB_spin_flip_adapted，
# 以及它们依赖的 jl/_network.jl 多组 OTF 兜底 / C++ |vt|≠1 内核。
# ===========================================================================

# ---------------------------------------------------------------------------
# 覆盖 ansatz.jl 中的 unique_operator_pool：
# 原版只按完全相等去重，UCCGSD 中 [vir...,occ...] 与其共轭 [occ...,vir...]
# 生成的算符互为精确相反数（X 与 -X），却都被保留，导致池子翻倍。
# 这里在去重前做符号规范化（首个非零系数取正），使 X 与 -X 合并为同一个算符。
# ---------------------------------------------------------------------------
function unique_operator_pool(pool::Vector{BinaryQubitAABB{Ti,Tv,K,V}}) where {Ti,Tv,K,V}
    len = length(pool)
    index_map = Dict{eltype(pool),Int}()
    sizehint!(index_map, len)
    unique_indices = Array{Int,1}(undef, len)

    To = eltype(pool)

    canon = op -> begin
        op2 = op
        for c in op.cs
            if !iszero(c)
                if real(c) < 0 || (iszero(real(c)) && imag(c) < 0)
                    op2 = -op
                end
                break
            end
        end
        return op2
    end

    count = 0
    for (i, op) in enumerate(pool)
        (op == zero(To) || op == one(To)) && continue
        c = canon(op)
        if !haskey(index_map, c)
            count += 1
            index_map[c] = i
            unique_indices[count] = i
        end
    end

    resize!(unique_indices, count)

    return pool[unique_indices]
end


# ---------------------------------------------------------------------------
# GSD 算符分类约化
#
# 对 generalized GSD 池（kernel(...; generalize=true)），一个激发 a 的指标
# 前一半是 creation、后一半是 annihilation。ne = na + nb 是占据自旋轨道
# 的边界，因此 count(a .< ne) 表示有多少个指标落在占据块：
#   单激发 (length(a)==2) : T11 (1 个占据) / T10 (0 个占据)
#   双激发 (length(a)==4) : T22 (2 个) / T21 (1 或 3 个) / T20 (0 或 4 个)
# 常用 keep = ["T21","T22"] 可去掉纯 hole-hole / particle-particle 双激发。
# ---------------------------------------------------------------------------
function reduce_gsd_class(orbs::Orbitals, ne::Int; keep::Vector{String}=["T10","T11","T20","T21","T22"])
    valid = Set(["T10", "T11", "T20", "T21", "T22"])
    for c in keep
        c in valid || throw(ArgumentError("Unknown class: $(c)"))
    end

    cls = a -> begin
        n_occ = count(a .< ne)
        if length(a) == 2
            return n_occ == 1 ? "T11" : "T10"
        else
            return n_occ == 2 ? "T22" : (n_occ in (1, 3) ? "T21" : "T20")
        end
    end

    keep_set = Set(keep)
    orbs.spin_based = [a for a in orbs.spin_based if cls(a) in keep_set]
    return orbs
end


# 在 T21 类中，剔除涉及最底层 ncore 个 core 轨道或最高 ntop 个虚轨道的项。
# 单激发全部保留；只有 n_occ ∈ {1,3}（即 T21）的项会被 core/top 条件过滤。
function filter_gsd_T21(orbs::Orbitals, ne::Int, nsp::Int; ncore::Int=0, ntop::Int=0)
    core_low = 2 * ncore
    top_low  = nsp - 2 * ntop

    orbs.spin_based = filter(orbs.spin_based) do a
        length(a) == 2 && return true
        n_occ = count(a .< ne)
        (n_occ == 1 || n_occ == 3) || return true
        ncore > 0 && any(i -> i < core_low, a) && return false
        ntop  > 0 && any(i -> i >= top_low, a) && return false
        return true
    end
    return orbs
end


# ---------------------------------------------------------------------------
# 叠层 ansatz 的算子下标
#
# Binsim_10.27 的 run_vqe 直接接受 idxs::Vector{Int64}，所以叠层不需要复制
# 算符池，只要把下标序列重复即可：
#   order = "abab" : A B | A B | ...   （整层重复）
#   order = "aabb" : A A | B B | ...   （每个算符连续重复）
# 等价于 old_BinSim 的 layered_pool + run_vqe_layered，但不复制 pool。
# ---------------------------------------------------------------------------
function layered_idxs(nops::Int, nlayers::Int; order::String="abab")
    nlayers <= 1 && return collect(Int64, 1:nops)
    order in ("abab", "aabb") || throw(ArgumentError("Unknown order: $(order)"))

    if order == "abab"
        return repeat(collect(Int64, 1:nops), nlayers)
    else
        return Int64[i for i in 1:nops for _ in 1:nlayers]
    end
end
