# calculations —— old_BinSim/_try 算例的 Binsim_10.27 版本

所有脚本都通过 `common.jl` 加载 `jl/binsim.jl`，并统一使用新接口：

```bash
cd Binsim_10.27
julia --project=. calculations/water_reduce_gsd.jl
```

## 与 old_BinSim/_try 的对应关系

| old_BinSim/_try | 本目录 |
|---|---|
| `water_reduceGSD.jl` | `water_reduce_gsd.jl` |
| `h2o_sto6g.jl`（git HEAD） | `h2o_sto6g.jl` |
| `Ne.jl`（git HEAD） | `ne_reduce_gsd.jl` |
| `backup/check_frozencore/frozen4_n2.jl` | `n2_frozen_core_scan.jl` |
| `backup/multi_layer/water_sto3g/water_layer_*.jl` | `water_layered.jl`（abab/aabb 在同一脚本内） |
| `backup/multi_layer/old/n2_layer.jl` | `n2_layered.jl` |
| `backup/old/_scan_n2.jl` | `n2_scan_hforb.jl` |
| `backup/old/deepseek_uccsd.jl` | `n2_uccsd.jl` |
| `backup/check_symm/runpyscf_n2hf_diffsym.py` | `check_symm/runpyscf_n2hf_diffsym.py` |
| `backup/check_symm/{all_c1-hforb,all_d2h-hforb,corresponding_hforb}/n2.jl` | `check_symm/{c1_hforb,d2h_hforb,corresponding_hforb}.jl` |
| `backup/align_orbital/v1.py,v2_*,v3_yeah,v4` | `align_orbital/`（原样复制） |
| `print_orbsym_n2.py` | `align_orbital/print_orbsym_n2.py` |
| `for_spin_adapted/` | **不移植**（自旋匹配） |

## 注意事项

1. **自旋匹配未移植**：`kernel(...; spin_adapted=true)`、`SpinAdaptedFEB/QEB`、
   `FEB_spin_flip_adapted`、`reduce_spin_flip!` 以及 `_network.jl` 的多组 OTF
   兜底都不在 Binsim_10.27 中；相应的 `for_spin_adapted/` 与
   `water_spinflip*.out` 也没有对应算例。

2. **没有 `run_exact_vqe`**：Binsim_10.27 删除了 `method.jl`（含
   `krylov_expmv!` / `run_exact_vqe`），因此旧脚本里的 "Exact UCCSD vs
   Trotterized" 对比在这里只剩 Trotterized（+ FCI 参考值）。需要精确 UCC
   时再单独移植 `method.jl` 的对应部分。

3. **`hforb` 语义**：`Mole.hforb` 现在是**全基组**的 `mf.mo_coeff`
   （`nao × nao`），即使 `frozen_core > 0` 也不裁列，因此可以直接作为下一条
   几何的 RHF 初猜。需要活动空间轨道时用 `mole.hforb[:, ncore+1:end]`。

4. **缓存路径**：`molecule_from_geometry` 的缓存是
   `jld2file/<save_name>-<basis_key>.jld2`；`build(mole)` 仍是
   `jld2file/<name>-<ratio>-<basis>.jld2`。两者都不再使用旧的
   `jld2file/<basis>/` 子目录。

5. **`align_orbital` 的已知问题**：`v*.py` 用 `h5py` 写 `.jld2`，并且写到
   脚本目录下的 `jld2file/<basis>/`；JLD2.jl 未必能读 h5py 生成的文件，路径
   也和 `Binsim_10.27/jld2file/` 不一致。`v3_yeah/n2.jl` 与 `v4/n2.jl` 在
   `read_jld2=true` 读不到时会退回现场 PySCF（此时并没有用上对齐轨道）。
   如果要让它们真正串起来，需要把保存端改成 Julia 可读的 JLD2（或 Python
   存 `.npz` + Julia 转换）。

6. **重算成本**：`water_reduce_gsd.jl` / `ne_reduce_gsd.jl` / `water_layered.jl`
   的 VQE 参数规模与 old_BinSim 相同（几十到几千），完整跑完可能较慢；
   `n2_*` 系列是快速验证用的。
