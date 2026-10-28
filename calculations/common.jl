# ===========================================================================
# calculation 公共工具
#
# 所有算例都从这里 include binsim.jl，避免每个脚本各写一份环境变量和
# “建 OTF + 取 HF + 跑 VQE”的样板代码。
# ===========================================================================

ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "1")
ENV["OMP_PROC_BIND"] = get(ENV, "OMP_PROC_BIND", "close")
ENV["OMP_PLACES"] = get(ENV, "OMP_PLACES", "cores")

include(joinpath(@__DIR__, "..", "jl", "binsim.jl"))


# HF 参考向量的稀疏表示（Binsim_10.27 的 run_vqe 需要 v0_idxs / v0_vals）
function hf_sparse(basis::BasisManager)
    lv = get_hf(basis)
    idxs = findall(x -> x != 0, lv)
    vals = lv[idxs]
    return idxs, vals
end


# 单层/多层 Trotterized VQE。layers 通过 idxs 重复实现：
#   order = "abab" : 整层重复；order = "aabb" : 每个算符连续重复。
function run_pool_vqe(basis::BasisManager, ham, pool, e_ref::Float64;
    label::String="", nlayers::Int=1, order::String="abab", verbose::Int=1,
)
    v0_idxs, v0_vals = hf_sparse(basis)
    lv = get_hf(basis)
    rv = zeros(Float64, basis.dim)

    funcs = OTF_Functions(basis, ham, pool)
    idxs = layered_idxs(length(pool), nlayers; order=order)
    x0 = zeros(Float64, length(idxs))

    println("\n===== $(label): pool size = $(length(pool)), VQE params = $(length(idxs)) =====")
    e, _, _ = run_vqe(funcs, lv, rv, v0_idxs, v0_vals, e_ref, x0, idxs,
                      VQE_OPTIONS(verbose=verbose))
    println("$(label) energy = $(e)\n")
    return e
end


# FCI 参考能量（注意 run_fci 会用到初始向量，传 copy 更安全）
function fci_reference(basis::BasisManager, ham)
    return run_fci(basis, ham, copy(get_hf(basis)))
end
