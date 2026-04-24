# CATCCOS DGemm组件

CATCCOS的分层结构如下所示(以AllGather Matmul为例)，这些组件针对数据类型、数据排布和数学指令进行特化。


| API 层级             | API 类 和/或 函数 名称                   |
| ---                  | ---                                               |
| Kernel               | `Catccos::DGemm::Kernel::AllGatherMatmul`            |
| Block           | `Catlass::Gemm::Block::BlockMmad` <br /> `Catccos::Comm::Block::CommBlock` <br /> `Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh` <br /> `Catccos::Comm::Block::BlockCommSwizzle` <br /> |
| Tile | `TileRemoteCopy` <br /> |
| Basic                 | `AscendC::DataCopy` |

在CATCCOS中，我们为每个算子定义Host和Device两个模块。Host模块主要负责**算子执行的输入输出处理**，包括Device地址空间申请、输入数据构造、输出结果保存等功能。Device模块则负责汇总算子各个入参，然后**调用Kernel层级的API执行算子**。

具体可见CATCCOS的示例中[examples/allgather_matmul](../examples/allgather_matmul)，如下文摘录所示。
```c++
// examples/allgather_matmul/allgather_matmul_host.h
// 根据shape申请Device地址空间，如果是精度测试则通过dataFile将输入矩阵拷贝到申请的空间，如果是性能测试则将申请的空间全部赋值为1
void AllocateDevicePtrAGMM(KernelParams &params, uint32_t m, uint32_t k, uint32_t n, uint32_t rankId, uint32_t rankSize, std::string dataFile = "");

// 算子执行结束后，将Device空间计算结果拷贝到Host侧再保存到本地，用于结果验证。
void WriteResultAGMM(const KernelParams &params, uint32_t m, uint32_t n, uint32_t rankSize, std::string dataFile);
```

Device API调用说明

```c++
// examples/allgather_matmul/allgather_matmul_device.h
// Kernel API接口入参说明。
template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC
>
CATLASS_GLOBAL
void AllGatherMatmul(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR symmetricPtr, CocTilingParams cocTiling
);

// 第一步: 创建block层mmad
// 参数
using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<true>;
using L1TileShape = Catlass::GemmShape<M0, N0, K0>;
using L0TileShape = Catlass::GemmShape<M0, N0, 64>;

using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
using BlockMmad = Catlass::Gemm::Block::BlockMmad<
    MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType
>;

// 第二步：指定计算时的数据走位方式
using BlockMmadScheduler = Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;

// 第三步: 创建block层AllGather
// 参数
using RemoteSrcType = AType;
using RemoteDstType = AType;
using CopyDirect = Catccos::detail::CopyDirect;
// 指定数据搬运的方向和方式
using CopyTransport = Catccos::detail::CopyTransport;
using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Put, CopyTransport::Mte>;
using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

using CommDispatchPolicy = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
using BlockComm = Comm::Block::CommBlock<
    CommDispatchPolicy,
    RemoteSrcType, RemoteDstType,
    void,
    TileRemoteCopy, TileScheduler
>;

// 第四步：指定通信时的数据搬运策略
using BlockCommScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;

// 第五步：在kernel层将mmad和AllGather组合到一起
using AllGatherMatmulKernel = DGemm::Kernel::AllGatherMatmul<
    BlockMmad,
    BlockComm,
    BlockMmadScheduler,
    BlockCommScheduler,
    WORKSPACE_STAGES
>;

// 第六步：构造AllGatherMatmulKernel的Params
typename TileRemoteCopy::Params tileParams {
    commTileShape
};

typename BlockComm::Params blockParams {
    commBlockShape,
    tileParams
};

typename BlockCommScheduler::Params swizzleParams {
    commCoreSplit
};

typename AllGatherMatmulKernel::Params params {
    problemShape,
    rank, rankSize,
    commInterval,
    gmA, layoutA,
    gmB, layoutB,
    gmC, layoutC,
    symmetricPtr,
    blockParams,
    swizzleParams
};

// 第七步：实例化一个Kernel，并执行该算子
AllGatherMatmulKernel matmulCommKernel;
matmulCommKernel(params);
```

## Kernel API

Kernel层API是设备侧调用的入口，也是融合连续矩阵乘、通信数据搬运或其他操作的组合点。代码路径在include/catccos/dgemm/kernel下

```cpp
namespace Catccos::DGemm::Kernel {
template <
    class BlockMmad_,
    class BlockComm_,
    class BlockMmadScheduler_,
    class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class AllGatherMatmul;
} // namespace Catccos::DGemm::Kernel
```

## Block API

Block API包括“矩阵乘累加”和“通信算子数据搬运”两部分，其中矩阵计算使用catlass提供的BlockMmad接口，block层指定的矩阵分块计算策略在include/catccos/dgemm/block下定义。通信算子的数据搬运接口和搬运策略代码实现在include/catccos/comm/block下，支持远端读写及本地读写操作。

### 通信数据搬运
```c++
// include/catccos/comm/block/comm_block_remote_copy.hpp
template <
    uint32_t UB_STAGES_,
    bool IsDynamic_,
    class SrcType_,
    class DstType_,
    class BlockShape_,
    class TileRemoteCopy_,
    class TileSwizzle_
>
class CommBlock <
    AtlasA2CommRemoteCopy<UB_STAGES_, IsDynamic_>,
    SrcType_,
    DstType_,
    BlockShape_,
    TileRemoteCopy_,
    TileSwizzle_
>;
```

## Tile API

Tile粒度的数据搬运实现在include/catccos/comm/tile下，支持put和get两种数据搬运方向(CopyDirect设置)、mte和rdma两种数据搬运模式(CopyTransport设置)。

```cpp
template <
    class ArchTag,
    bool IsDynamic_, // tileshape参数是否为动态的
    class SrcType_,
    class DstType_,
    class TileShape_,
    detail::CopyDirect CopyDirect_,
    detail::CopyTransport CopyTransport_
>
struct TileRemoteCopy;
```

## Basic API

Basic层级API封装了实际的硬件指令调用，这些指令加速了MMAD或数据拷贝操作，实现了对硬件能力的抽象，开放了芯片能力，保证了完备性和兼容性，其中ISASI类API，不保证跨硬件版本兼容。