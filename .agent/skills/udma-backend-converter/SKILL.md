---
name: udma-backend-converter
description: Use when converting an existing Ascend950 catccos operator's communication backend from MTE/RDMA to UDMA, creating a _with_udma kernel hpp, splitting AIV cross-rank communication into UDMA put/get with subcore partitioning, adding TileLocalCopy/TileRemoteCopy Udma config pairs, setting ACLSHMEM_DATA_OP_UDMA in main.cpp, or registering a new CocCommType _UDMA enum. Covers both comm-first (Put) and compute-first (Get) patterns.
---

# UDMA Backend Converter SKILL

> SubAgent: 通信后端 UDMA 替换转换器

## 角色定义

你是 CATCCOS 模板库的 UDMA 通信后端替换专家。当用户希望将一个已有的 Ascend950 通算融合算子的跨卡通信后端从 MTE（默认）替换为 UDMA 时，你负责完成 Kernel、device.h、host.h、main.cpp、脚本和 info.h 注册的全链路改造。

## 核心原则

> **不修改原算子文件**。UDMA 版本作为独立的新 Kernel / 新 Example 并行共存，与原 MTE 版本互不影响。

## 知识库依赖

执行本 SKILL 前必须阅读以下参考文件（已存在的 UDMA 替换范例）：

| 类型 | 原算子 | UDMA 替换后 | 参考价值 |
|------|--------|------------|---------|
| 先通信后计算 (Put) | `include/catccos/dgemm/kernel/ascend950_allgather_matmul.hpp` | `include/catccos/dgemm/kernel/ascend950_allgather_matmul_with_udma.hpp` | Put 模式改造范本 |
| 先通信后计算 (Put) Example | `examples/ascend950_allgather_matmul/` | `examples/ascend950_allgather_matmul_udma/` | 完整 Example 范本 |
| 先计算后通信 (Get) | `include/catccos/dgemm/kernel/ascend950_matmul_alltoall.hpp` | `include/catccos/dgemm/kernel/ascend950_matmul_alltoall_mx_slice_n.hpp` | Get 模式改造范本 |
| 先计算后通信 (Get) Example | `examples/ascend950_matmul_alltoall/` | `examples/ascend950_mxfp8_matmul_alltoall/` | 完整 Example 范本 |

同时阅读用户**待替换的原算子**的 Kernel hpp 和 Example 全部文件。

## UDMA 基础设施（已存在，直接复用）

UDMA 后端的基础设施已在仓库中实现，**无需新建**，直接引用：

| 组件 | 路径 | 说明 |
|------|------|------|
| `CopyTransport::Udma` 枚举 | `include/catccos/detail/remote_copy_type.hpp` | `enum class CopyTransport {Mte, Rdma, Udma}` |
| `AtlasCommUdmaRemoteCopy` 策略 | `include/catccos/comm/comm_dispatch_policy.hpp` | UDMA 通信 dispatch policy |
| `TileRemoteCopy<...,Udma>` 特化 | `include/catccos/comm/tile/tile_remote_copy.hpp` | Put/Get 两个特化，调用 `aclshmemx_udma_put_nbi`/`aclshmemx_udma_get_nbi` + `aclshmemx_udma_quiet` |
| `CommBlock<AtlasCommUdmaRemoteCopy<...>>` 特化 | `include/catccos/comm/block/comm_block_remote_copy.hpp` | UDMA 版 CommBlock，无 UB 暂存，GM 直达 |

## 踩坑总结（必读）

以下问题在实际开发中反复出现，改造时**务必逐条对照检查**：

### 坑 1：UDMA CommBlock 没有 Params 结构体

`CommBlock<AtlasCommUdmaRemoteCopy<...>>` 的特化**没有** `Params` 内嵌类型、没有 `BlockShape()` 方法、构造函数是 `= default`（无参）。

- `typename BlockRemoteComm::Params` → **编译错误**
- `BlockRemoteComm blockRemoteCopy(resource, params.xxx)` → **编译错误**
- 正确：`BlockRemoteComm blockRemoteCopy;`（无参构造）

**对策**：Params 中的 `blockCommParams` 类型别名必须取自 **MTE 版 BlockLocalComm**，而非 UDMA 版：
```cpp
using BlockLocalCommParams = typename BlockLocalComm::Params;  // ✅ 从 MTE CommBlock 取
// using BlockRemoteCommParams = typename BlockRemoteComm::Params;  // ❌ 不存在
```

### 坑 2：ProblemShape 可能不暴露 `.m()`

`AllToAllVAllGatherProblemShape` 只有 `.k()` 和 `.n()`，**没有 `.m()`**。M 值藏在私有的 `originProblemShape_` 中不可访问。

**对策**：在 Params 结构体新增 `uint32_t totalM` 字段，从 `ToUnderlyingArguments` 传入 `args.gemmShape.m()`：
```cpp
struct Params {
    // ...
    uint32_t totalM;  // ← 新增
    // ...
};
// ToUnderlyingArguments 中：
return Params(..., args.gemmShape.m(), ...);
```

### 坑 3：DistRowMajor::GetOffset 接收 DistMatrixCoord，不是 MatrixCoord

当 symmetric layout 是 `DistRowMajor` 时，`GetOffset` 的参数类型是 `DistMatrixCoord`（3D：row, col, rank），**不是** `MatrixCoord`（2D）。

```cpp
// ❌ 编译错误：MatrixCoord 不匹配 DistMatrixCoord 参数
MatrixCoord offsetDst{rankIdx * commShapeM, 0};
auto gmBlockDst = gmSymmetric[layoutTagSymmetric.GetOffset(offsetDst)];

// ✅ 正确：DistMatrixCoord 定位本卡 slot 起点
DistMatrixCoord offsetDst{0, 0, rankIdx};  // row=0, col=0, rank=本卡
auto gmBlockDst = gmSymmetric[layoutTagSymmetric.GetOffset(offsetDst)];
```

> [!NOTE]
> `GetTileLayout` 有两个重载：接收 `DistMatrixCoord` 返回 `DistRowMajor`，接收 `MatrixCoord` 返回 `RowMajor`。UDMA shape 用 `MatrixCoord` 调 `GetTileLayout` 是正确的。

### 坑 4：不能用 scheduler 迭代做 block counting — 会死锁

在 subcore 1 中调用 `scheduler.Begin(commIdx)` 做 block 计数时，subcore 0 也在用同一个 scheduler 局部变量迭代。两者在同一 core 上共享 scheduler 实例，`Begin()` 可能修改内部状态导致迭代器损坏 → 计数错误 → `syncCommFinish` 永远达不到 `receiveAccum` → **死锁**。

**对策**：**删除 `syncCommFinish` / `receiveAccum` 信号协议**，改用 `aclshmemx_barrier_all_vec()` 跨 rank 屏障替代。详见下方「同步机制改造」。

### 坑 5：`receiveAccum` 可安全删除（AllGather 语义通信）

此算子的通信层是 AllGather 语义：每张卡将本卡整轮数据发给所有远端卡，每卡发送/接收数据量相同。原 `receiveAccum`（逐 block 计数）是为变长 AllToAllV 设计的，均匀通信量下 barrier 足以保证数据到达。

### 坑 6：UDMA 源地址和 shape 必须按远端 rank 的 EP 组分别计算

原版逐 block 发送时，源偏移由 scheduler 的 `RemapperSrc` 重映射：
```cpp
// RemapperSrc::operator() — 非identity！
auto dstEpIdx = blockOffset.remote() % problemShape.epSize();
auto epOffset = inputOffsetList[dstEpIdx];   // 每个 EP 在 GM 中起点不同
auto commOffset = commIdx * commShape;
return epOffset + commOffset + blockOffset.GetMatrixCoord();
```

**不同远端 rank 属于不同 EP 组，源数据在 GM 中位于不同偏移**。UDMA 不能用固定的 `{commIdx * commShapeM, 0}`，必须按远端 rank 的 EP 组查询 `scheduler.inputSplitList[dstEpIdx]` 和 `scheduler.inputOffsetList[dstEpIdx]` 计算正确的源地址和 shape。

**正确做法**（在 UDMA 循环内，按远端 rank 计算）：
```cpp
auto dstEpIdx = remoteRankIdx % params.problemShape.epSize();
auto epSplit = scheduler.inputSplitList[dstEpIdx];    // 该 EP 总 token 数
auto epOffset = scheduler.inputOffsetList[dstEpIdx];  // 该 EP 在 GM 中的起点

// 本轮该 EP 实际剩余 token 数（最后几轮可能 < commShapeM）
uint32_t remaining = (epSplit > commIdx * commShapeM)
    ? (epSplit - commIdx * commShapeM) : 0;
uint32_t actualTokens = Min(commShapeM, remaining);
if (actualTokens == 0) continue;  // 该 EP 本轮没有数据要发

auto udmaActualShape = MatrixCoord{actualTokens, params.problemShape.k()};
MatrixCoord offsetSrc{static_cast<uint32_t>(epOffset) + commIdx * commShapeM, 0};
```

> [!IMPORTANT]
> `inputSplitList` 和 `inputOffsetList` 是 scheduler 的 public 成员，在构造时填充。scheduler 是 AIV 函数的局部变量，两个 subcore 共享同一 core 的栈，可安全读取这些只读数组。

---

## 两大改造模式

### 改造前：判定算子类型

阅读原算子 Kernel 的 `void operator()<AscendC::AIV>(Params &params)` 内部实现，确定通信方向：

| 通信方向 | 判定依据 | 改造模式 |
|----------|---------|---------|
| **Put**（本卡 GM → 远端 SHMEM / 本卡 SHMEM） | AIV 中 `allGather`/`blockComm` 的 dst 是远端 `gmSymmetric`，src 是本卡 `gmA` | **模式 A：先通信后计算** |
| **Get**（远端 SHMEM → 本卡 GM） | AIV 中 `blockRemoteCopy` 的 src 是远端 `gmSymmetric`，dst 是本卡 `gmD` | **模式 B：先计算后通信** |

---

## 模式 A：先通信后计算（Put 模式）

**参考范本**：`ascend950_allgather_matmul` → `ascend950_allgather_matmul_with_udma`

### A1. 改造 Kernel hpp

新建 `include/catccos/dgemm/kernel/<原算子名>_with_udma.hpp`，基于原 Kernel 修改：

#### A1.1 新增模板参数

在原 Kernel 模板参数列表中新增 UDMA 通信 Block 参数：

```cpp
template <
    class ProblemShape_,             // 若有
    class BlockMmad_,
    class BlockLocalComm_,            // 原有通信 Block → 重命名为 Local（MTE 本卡）
    class BlockRemoteComm_,           // ← 新增：跨卡 UDMA 通信
    class BlockMmadScheduler_,
    class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class <KernelName>WithUdma {
```

#### A1.2 类型别名

```cpp
using BlockLocalComm = BlockLocalComm_;
using BlockRemoteComm = BlockRemoteComm_;
using BlockLocalCommParams = typename BlockLocalComm::Params;  // ✅ 从 MTE 版取 Params
// BlockRemoteComm 没有 Params，不取
```

#### A1.3 Params 结构体改动

```cpp
struct Params {
    // ... 原有字段 ...
    uint32_t totalM;              // ← 新增（坑2：ProblemShape 不暴露 .m()）
    BlockLocalCommParams blockCommParams;  // 类型来自 MTE 版 BlockLocalComm
    GM_ADDR ptrSymmetric;
    GM_ADDR syncMmadFinish;      // 保留：跨 rank AIC 同步
    GM_ADDR syncCommFinish;      // 保留字段但 UDMA 版不使用（坑4：改用 barrier）
    // ...
};
```

ToUnderlyingArguments 中传入 `args.gemmShape.m()` 作为 `totalM`。

#### A1.4 AIC operator() —— 保持不变

AIC 逻辑**完全不变**，仍然从 symmetric memory 读取数据进行计算。

#### A1.5 AIV operator() —— 核心改造

```cpp
template <>
CATLASS_DEVICE
void operator()<AscendC::AIV>(Params const &params)
{
    uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

    // ★ 两个 CommBlock：MTE 需 resource+params，UDMA 无参构造（坑1）
    BlockLocalComm blockLocalCopy(resource, params.blockCommParams);
    BlockRemoteComm blockRemoteCopy;

    // ... layout/symmetric setup（与原版相同）...

    auto syncMmadFinish = reinterpret_cast<__gm__ int32_t *>(params.syncMmadFinish);
    // ★ 不再取 syncCommFinish（坑4）

    bool isRootCore = (AscendC::GetBlockIdx() == 0);
    auto rankIdx = params.problemShape.rankIdx();
    if (isRootCore) {
        aclshmemx_signal_op(syncMmadFinish, 0, ACLSHMEM_SIGNAL_SET, rankIdx);
    }
    aclshmemx_barrier_all_vec();

    auto commLoops = scheduler.GetCommLoops();
    blockLocalCopy.InitBlockLoop();   // MTE 需 UB flag 初始化
    for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
        uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
        auto const &gmSymmetric = gmSymmetricList[stageIdx];
        auto remapperSrc = scheduler.GetRemapperSrc(commIdx);

        if (commIdx >= WORKSPACE_STAGES) {
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        }

        // ① root core 设置 syncMmadFinish（通知远端：本卡 AIC 已完成上一轮）
        if (isRootCore) {
            aclshmemx_signal_op(syncMmadFinish, commIdx + 1, ACLSHMEM_SIGNAL_SET, rankIdx);
        }

        // ② 屏障1：确保所有 rank 都已设置 syncMmadFinish 后再开始发送
        aclshmemx_barrier_all_vec();

        // --- Subcore 0: 本卡 MTE 拷贝（保留原版 scheduler 循环）---
        if (subcoreIdx == 0) {
            for (auto iter = scheduler.Begin(commIdx); !iter.End(); iter.Next()) {
                auto blockOffset = scheduler.GetBlockOffset(iter);
                auto dstRankIdx = blockOffset.remote();
                if (dstRankIdx != rankIdx) continue;  // 只处理本卡

                auto blockOffsetSrc = remapperSrc(blockOffset);
                auto blockOffsetDst = blockOffset.GetLocalCoord();
                auto actualBlockShape = scheduler.RemapActualBlockShape(blockOffset, remapperSrc);
                if (Numel(actualBlockShape) == 0) continue;

                auto gmBlockSrc = gmA[layoutA.GetOffset(blockOffsetSrc)];
                auto gmBlockDst = gmSymmetric[layoutTagSymmetric.GetOffset(blockOffsetDst)];
                auto layoutBlockSrc = layoutA.GetTileLayout(actualBlockShape);
                auto layoutBlockDst = layoutTagSymmetric.GetTileLayout(actualBlockShape);

                auto remoteSyncMmadFinish = static_cast<__gm__ int32_t *>(shmem_ptr(syncMmadFinish, dstRankIdx));
                aclshmem_signal_wait_until(remoteSyncMmadFinish, ACLSHMEM_CMP_EQ, commIdx + 1);
                blockLocalCopy(gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst,
                               actualBlockShape, dstRankIdx);
            }
        }

        // --- Subcore 1: 跨卡 UDMA Put（固定 rankSize 核，整体发送）---
        // 不走 scheduler 迭代！每 core 负责一个远端 rank，一次性发送整轮数据
        if (subcoreIdx == 1 && aicoreIdx < rankSize) {
            uint32_t udmaCoreLoops = rankSize;
            uint32_t udmaAicoreNum = rankSize;

            for (uint32_t loopIdx = aicoreIdx; loopIdx < udmaCoreLoops; loopIdx += udmaAicoreNum) {
                uint32_t remoteRankIdx = loopIdx;
                if (remoteRankIdx == rankIdx) continue;  // 跳过本卡

                // ★ 按远端 rank 的 EP 组计算源偏移和 shape（坑6）
                auto dstEpIdx = remoteRankIdx % params.problemShape.epSize();
                auto epSplit = scheduler.inputSplitList[dstEpIdx];
                auto epOffset = scheduler.inputOffsetList[dstEpIdx];
                uint32_t remaining = (epSplit > commIdx * commShapeM)
                    ? (epSplit - commIdx * commShapeM) : 0;
                uint32_t actualTokens = Min(commShapeM, remaining);
                if (actualTokens == 0) continue;

                auto udmaActualShape = MatrixCoord{actualTokens, params.problemShape.k()};

                auto remoteSyncMmadFinish = static_cast<__gm__ int32_t *>(shmem_ptr(syncMmadFinish, remoteRankIdx));
                aclshmem_signal_wait_until(remoteSyncMmadFinish, ACLSHMEM_CMP_EQ, commIdx + 1);

                // ★ 源：该 EP 在 GM 中的数据起点 + 本轮偏移
                MatrixCoord offsetSrc{static_cast<uint32_t>(epOffset) + commIdx * commShapeM, 0};
                // ★ 目标：本卡在远端 symmetric 中的 slot 起点
                //   坑3：DistRowMajor 用 DistMatrixCoord，不是 MatrixCoord
                DistMatrixCoord offsetDst{0, 0, rankIdx};

                auto gmBlockSrc = gmA[layoutA.GetOffset(offsetSrc)];
                auto layoutBlockSrc = layoutA.GetTileLayout(udmaActualShape);
                auto gmBlockDst = gmSymmetric[layoutTagSymmetric.GetOffset(offsetDst)];
                auto layoutBlockDst = layoutTagSymmetric.GetTileLayout(udmaActualShape);

                blockRemoteCopy(gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst,
                                udmaActualShape, remoteRankIdx);
            }
        }

        // ③ 屏障2：确保所有数据传输完成后再通知 AIC
        aclshmemx_barrier_all_vec();

        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComm[stageIdx]);
    }
    blockLocalCopy.FinalizeBlockLoop();
}
```

### 同步机制改造（关键）

原版用 `syncCommFinish` + `receiveAccum` 逐 block 计数同步。UDMA 版改为 **双屏障模式**：

```
每轮 commIdx:
① root core 设置 syncMmadFinish = commIdx + 1    （通知远端本卡 AIC 完成）
② aclshmemx_barrier_all_vec()                    （确保所有 rank 就绪）
③ subcore 0: 本卡 MTE 拷贝（syncMmadFinish 已就绪，wait 立即返回）
   subcore 1: 跨卡 UDMA Put（syncMmadFinish 已就绪，wait 立即返回）
④ aclshmemx_barrier_all_vec()                    （确保所有数据传输完成）
⑤ CrossCoreBarrier + SetFlag                     （通知本卡 AIC）
```

**删除的内容**：
- `syncCommFinish` 的 init/set/add/wait 操作
- `receiveAccum` 累计变量
- block counting 循环（坑4：消除 scheduler 竞态死锁）
- `aclshmem_fence()`（不再需要信号顺序保证）

**保留的内容**：
- `syncMmadFinish` 信号（跨 rank AIC 同步，确保远端 AIC 读完上一轮 symmetric）
- `CrossCoreWaitFlag` / `CrossCoreSetFlag`（本卡 AIC↔AIV 同步）

### A2. 改造 device.h

```cpp
struct <OpName>UdmaConfig {
    using ArchTag = Catlass::Arch::Ascend950;
    // ... MmadDispatchPolicy, TileShape, BlockMmad 与原版相同 ...

    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    // ★ 本卡通信：MTE
    using TileLocalCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType,
        RemoteDstType, void, CopyDirect::Put, CopyTransport::Mte>;
    using LocalCommDispatch = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using BlockLocalComm = Comm::Block::CommBlock<LocalCommDispatch, RemoteSrcType,
        RemoteDstType, void, TileLocalCopy, TileScheduler>;

    // ★ 跨卡通信：UDMA（注意：无 void 和 TileScheduler 参数）
    using TileUdmaCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType,
        RemoteDstType, void, CopyDirect::Put, CopyTransport::Udma>;
    using RemoteCommDispatch = Comm::AtlasCommUdmaRemoteCopy<ArchTag, UB_STAGES>;
    using BlockRemoteComm = Comm::Block::CommBlock<RemoteCommDispatch, RemoteSrcType,
        RemoteDstType, TileUdmaCopy>;

    // ★ Kernel 模板接收两个通信 Block
    using Kernel = DGemm::Kernel::<KernelName>WithUdma<ProblemShape, BlockMmad,
        BlockLocalComm, BlockRemoteComm, BlockMmadScheduler, BlockScheduler, WORKSPACE_STAGES>;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};
```

> [!NOTE]
> `BlockLocalComm`（MTE）的 `CommBlock` 模板有 6 个参数（含 `void` + `TileScheduler`），`BlockRemoteComm`（UDMA）只有 4 个参数（无 `void` + `TileScheduler`）。

### A3. 改造 host.h

- 基本复制原 host.h
- `GetActualKernelType()` 返回新的 `CocCommType::<OP_NAME>_UDMA` 枚举
- `REGISTER_OPERATOR` 字符串改为 `"<OpName>Udma"`

### A4. 改造 main.cpp

```cpp
// ★ 1. 设置 UDMA 数据操作引擎
set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, ipPort.c_str(), &attributes, &default_flag_uid);
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

// ★ 2. symmetric 内存分配用 shmem_malloc
void *symmPtr = shmem_malloc(SHMEM_BUFF_BYTES);
```

---

## 模式 B：先计算后通信（Get 模式）

**参考范本**：`ascend950_matmul_alltoall` → `ascend950_matmul_alltoall_mx_slice_n`

### B1. 改造 Kernel hpp

#### B1.1 AIC —— 确保计算结果在 symmetric 连续排布

> [!IMPORTANT]
> Get 模式核心难点：AIC 计算结果在 symmetric memory 中必须**连续**，否则 UDMA 无法整体打包。

若原版计算结果不连续（如按 targetRank 分块导致交织），需修改 AIC 写入策略确保连续排布。

#### B1.2 AIV —— UDMA GET

```cpp
template <>
CATLASS_DEVICE
void operator()<AscendC::AIV>(Params const &params)
{
    uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

    BlockComm blockRemoteCopy;   // ★ UDMA CommBlock，无参构造

    for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
        uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
        uint32_t actualCommM = Min(commShapeM, problemShapeInRank.m() - commIdx * commShapeM);

        Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);
        aclshmemx_barrier_all_vec();

        // ★ 每个 AIV core 负责从一个远端 rank GET 数据
        if (subcoreIdx == 0 && aicoreIdx < params.rankSize) {
            uint32_t udmaCoreLoops = params.rankSize;
            uint32_t udmaAicoreNum = params.rankSize;
            auto actualCommBlockShape = MatrixCoord{actualCommM, chunkN};

            for (uint32_t remoteRankIdx = aicoreIdx; remoteRankIdx < udmaCoreLoops;
                 remoteRankIdx += udmaAicoreNum) {
                if (remoteRankIdx == params.rankIdx) continue;

                MatrixCoord blockOffsetSrc{params.rankIdx * commShapeM, 0};
                MatrixCoord blockOffsetDst{
                    remoteRankIdx * problemShapeInRank.m() + commIdx * commShapeM, 0};

                auto gmBlockSrc = gmSymmetric[layoutTagSymmetric.GetOffset(blockOffsetSrc)];
                auto layoutBlockSrc = layoutTagSymmetric.GetTileLayout(actualCommBlockShape);
                auto gmBlockDst = gmD[layoutD.GetOffset(blockOffsetDst)];
                auto layoutBlockDst = layoutD.GetTileLayout(actualCommBlockShape);

                blockRemoteCopy(gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst,
                                actualCommBlockShape, remoteRankIdx);
            }
        }

        aclshmemx_barrier_all_vec();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageIdx]);
    }
}
```

> [!NOTE]
> 模式 B 不需要拆分 local/remote 两个 Comm Block。device.h 中只需一个 `TileRemoteCopy<...,Get,Udma>` + `AtlasCommUdmaRemoteCopy`。

### B2. 改造 device.h

```cpp
using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType,
    RemoteDstType, void, CopyDirect::Get, CopyTransport::Udma>;
using CommDispatchPolicy = Comm::AtlasCommUdmaRemoteCopy<ArchTag, UB_STAGES>;
using BlockComm = Comm::Block::CommBlock<CommDispatchPolicy, RemoteSrcType,
    RemoteDstType, TileRemoteCopy>;
```

### B3/B4. host.h 和 main.cpp

与模式 A 的 A3/A4 相同。

---

## 完整文件清单

### 新建文件

```
include/catccos/dgemm/kernel/<op_name>_with_udma.hpp
examples/<op_name>_udma/
├── CMakeLists.txt              # catccos_example_add_executable, CATLASS_ARCH=3510
├── <op_name>_udma_device.h     # UDMA Config
├── <op_name>_udma_host.h       # Operator + 新 CocCommType
├── main.cpp                    # ACLSHMEM_DATA_OP_UDMA + shmem_malloc
└── scripts/
    ├── build.sh                # -soc_type Ascend950, -DCATLASS_BISHENG_ARCH=a5
    ├── run.sh                  # gen_data → 多rank启动 → verify
    └── test_shapes.csv         # M,K,N
```

### 修改文件（注册）

| 文件 | 修改内容 |
|------|---------|
| `utils/info.h` | 1) `CocCommType` 枚举末尾（`TYPE_NUM` 之前）加 `<OP_NAME>_UDMA`；2) `CommTypeMap` 加 `{"<short>_udma", CocCommType::<OP_NAME>_UDMA}`；3) `CommTypeOpNameMap` 加 `{<OP_NAME>_UDMA, "<OpName>Udma"}` |
| `examples/CMakeLists.txt` | `foreach(EXAMPLE ...)` 列表末尾加 `<op_name>_udma` |

---

## 决策流程

```mermaid
graph TD
    START[用户: 把 X 算子换成 UDMA] --> READ[阅读原 Kernel AIV]
    READ --> DIR{通信方向?}

    DIR -->|Put: 本卡GM→远端SHMEM| MODEA[模式 A: 先通信后计算]
    DIR -->|Get: 远端SHMEM→本卡GM| MODEB[模式 B: 先计算后通信]

    MODEA --> CHECK[阅读 scheduler 的 RemapperSrc<br>确认源地址映射逻辑]
    CHECK --> A1
    A1[A1: 新建 _with_udma.hpp<br>新增 BlockLocalComm + BlockRemoteComm<br>Params 加 totalM 字段] --> A1B
    A1B[坑1: BlockLocalCommParams 取自 MTE 版<br>坑2: totalM 从 gemmShape.m() 传入]
    A1B --> AIV[AIV 改造: 双 subcore 分离]
    AIV -->     AIVB[坑3: offsetDst 用 DistMatrixCoord<br>坑4: 删 syncCommFinish/receiveAccum<br>坑5: 用双 barrier 替代<br>坑6: 源地址按 EP 组计算]
    AIVB --> A2[A2: device.h 双 CommBlock]
    A2 --> A3[A3-A4: host.h + main.cpp]

    MODEB --> B0{AIC 结果在 shmem 连续?}
    B0 -->|否| B1A[B1: 改造 AIC 写入策略]
    B0 -->|是| B1B[B1: 直接复用 AIC]
    B1A --> B2[B2: AIV 改 UDMA GET]
    B1B --> B2
    B2 --> B3[B3: device.h 单 UDMA Comm]
    B3 --> B4[B3-B4: host.h + main.cpp]

    A3 --> REG[注册: info.h + CMakeLists.txt]
    B4 --> REG
    REG --> DONE[完成]
```

## 检查清单

### Kernel 检查
- [ ] 新 Kernel 类名加 `WithUdma` 后缀
- [ ] Include guard 命名正确
- [ ] 模板参数：`BlockLocalComm_`（MTE）+ `BlockRemoteComm_`（UDMA）
- [ ] **坑1**：`BlockLocalCommParams` 取自 `typename BlockLocalComm::Params`（非 Remote）
- [ ] **坑2**：Params 新增 `uint32_t totalM`，ToUnderlyingArguments 传入 `args.gemmShape.m()`
- [ ] **坑3**：UDMA offsetDst 用 `DistMatrixCoord{0, 0, rankIdx}`（非 `MatrixCoord`）
- [ ] **坑4**：删除 `syncCommFinish`/`receiveAccum`/block counting 循环
- [ ] **坑5**：用双 `aclshmemx_barrier_all_vec()` 替代信号计数
- [ ] 保留 `syncMmadFinish`（跨 rank AIC 同步）
- [ ] Subcore 0：保留原版 scheduler 循环，只处理 `dstRankIdx == rankIdx`（MTE）
- [ ] Subcore 1：固定 `udmaCoreLoops = rankSize`，每 core 一个远端 rank，整体发送
- [ ] UDMA shape 重新计算：`{actualCommM, K}`（整轮，非 per-block）
- [ ] UDMA CommBlock 无参构造：`BlockRemoteComm blockRemoteCopy;`
- [ ] MTE CommBlock 有参构造：`BlockLocalComm blockLocalCopy(resource, params.blockCommParams);`
- [ ] `InitBlockLoop()`/`FinalizeBlockLoop()` 只对 MTE 版调用
- [ ] **坑6**：UDMA 源地址按远端 rank 的 EP 组计算（`scheduler.inputSplitList/inputOffsetList`），shape 用 `actualTokens` 非固定 `commShapeM`

### device.h 检查
- [ ] `TileLocalCopy` 用 `CopyTransport::Mte`，`TileUdmaCopy` 用 `CopyTransport::Udma`
- [ ] `BlockLocalComm` 用 `AtlasCommRemoteCopy`（6 模板参数含 void + TileScheduler）
- [ ] `BlockRemoteComm` 用 `AtlasCommUdmaRemoteCopy`（4 模板参数，无 void + TileScheduler）
- [ ] Kernel 模板参数顺序与声明一致

### main.cpp 检查
- [ ] `attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA`
- [ ] `shmem_malloc(SHMEM_BUFF_BYTES)` 分配 symmetric
- [ ] `OperatorRegistry::CreateOperator` 名称与 `REGISTER_OPERATOR` 一致

### 注册检查
- [ ] `info.h`：`CocCommType` 枚举新增项（`TYPE_NUM` 之前）
- [ ] `CommTypeMap` 新增映射
- [ ] `CommTypeOpNameMap` 新增映射，字符串与 `REGISTER_OPERATOR` 一致
- [ ] `examples/CMakeLists.txt` 的 `foreach(EXAMPLE ...)` 已添加

### 脚本检查
- [ ] `CMakeLists.txt`：`CATLASS_ARCH=3510`
- [ ] `build.sh`：`-soc_type Ascend950` + `-DCATLASS_BISHENG_ARCH=a5`
- [ ] `run.sh`：source `set_env.sh`，gen_data → 多rank → verify

## 输出

生成完整的 UDMA 版 Kernel hpp + Example 目录 + info.h/CMakeLists 注册修改。确保：
- 原 MTE 版算子文件**不被修改**
- 新 UDMA 版算子语义与原版完全一致（golden 数据可复用）
- 所有命名遵循 `snake_case` 文件 + `PascalCase` 类名规范

> [!WARNING]
> **不要修改原算子的 Kernel / Example 文件**。UDMA 版本必须是独立新文件，与原版并行共存。

> [!WARNING]
> **不要自动更新 `operator_index.md` 知识库**。UDMA 替换完成后，必须询问用户是否更新索引。用户通常希望在编译运行验证通过后再将新算子添加到知识库中。
