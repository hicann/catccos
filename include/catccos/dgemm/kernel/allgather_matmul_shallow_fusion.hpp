#ifndef CATCCOS_DGEMM_KERNEL_ALLGATHER_MATMUL_SHALLOW_FUSION_HPP
#define CATCCOS_DGEMM_KERNEL_ALLGATHER_MATMUL_SHALLOW_FUSION_HPP

#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/comm/tile/tile_remote_copy_local_ub_to_shmem.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/optimized_matmul_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "shmem.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catccos::DGemm::Kernel {
using namespace Catlass;

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class PaddingB, class TensorA_>
class AllGatherMatmulShallowFusion {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;

    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;

    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;

    using BlockScheduler = BlockScheduler_;
    template <class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template <>
    struct LayoutHelper<void> {
        using type = void;
    };
    using LayoutA = typename TensorA_::Layout;
    using LayoutB = std::conditional_t<std::is_void_v<PaddingB>, LayoutWB, typename LayoutHelper<PaddingB>::type>;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWB;
        LayoutWB layoutWB;
        GM_ADDR ptrSymmetric;
        LayoutWA layoutWA;
        // 标志位地址
        GM_ADDR ptrSignal;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
            GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrWB_, LayoutWB layoutWB_, GM_ADDR ptrSymmetric_,
            LayoutWA layoutWA_, GM_ADDR ptrSignal_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              ptrWB(ptrWB_),
              layoutWB(layoutWB_),
              ptrSymmetric(ptrSymmetric_),
              layoutWA(layoutWA_),
              ptrSignal(ptrSignal_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint8_t* ptrA;
        LayoutA layoutA;
        uint8_t* ptrB;
        LayoutB layoutB;
        uint8_t* ptrC;
        LayoutC layoutC;
        uint8_t* ptrWB;
        LayoutWB layoutWB;
        uint8_t* ptrSymmetric;
        LayoutWA layoutWA;
        uint8_t* ptrSignal;
    };

    static bool CanImplement(const Arguments& args) { return true; }

    static size_t GetWorkspaceSize(const Arguments& args) { return 0; }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        Params params{args.problemShape, args.ptrA,  args.layoutA,  args.ptrB,         args.layoutB,  args.ptrC,
                      args.layoutC,      args.ptrWB, args.layoutWB, args.ptrSymmetric, args.layoutWA, args.ptrSignal};
        return params;
    }

    CATLASS_DEVICE
    AllGatherMatmulShallowFusion() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    // AIC: multiply the gathered A rows by the local B matrix.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrSymmetric);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);

        auto tensorA = tla::MakeTensor(gmA, params.layoutWA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutWB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        uint32_t rankId = aclshmem_my_pe();
        uint32_t rankSize = aclshmem_n_pes();

        // Wait until AIV has gathered every rank's A rows before starting GEMM.
        Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishAllGather);

        GemmCoord actualCocShape{params.problemShape.m() * rankSize, params.problemShape.n(), params.problemShape.k()};

        BlockScheduler matmulBlockScheduler(actualCocShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            auto tensorBlockA = GetTile(
                tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tensorBlockB = GetTile(
                tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            GemmCoord nextBlockCoord;
            GemmCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                nextBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockCoord);
            }

            auto nextTensorBlockA = GetTile(
                tensorA, tla::MakeCoord(nextBlockCoord.m() * L1_TILE_M, nextBlockCoord.k() * L1_TILE_K),
                tla::MakeShape(nextActualBlockShape.m(), nextActualBlockShape.k()));
            auto nextTensorBlockB = GetTile(
                tensorB, tla::MakeCoord(nextBlockCoord.k() * L1_TILE_K, nextBlockCoord.n() * L1_TILE_N),
                tla::MakeShape(nextActualBlockShape.k(), nextActualBlockShape.n()));

            // Compute block-scoped matrix multiply-add
            blockMmad(
                tensorBlockA, tensorBlockB, tensorBlockC, nextTensorBlockA, nextTensorBlockB, actualBlockShape,
                nextActualBlockShape, isFirstBlock, hasNextBlock);
        }
    }

    // AIV: prepare B and gather A rows into symmetric memory.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        aclshmemx_barrier_all_vec();
        if constexpr (!std::is_void_v<PaddingB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrWB));
            auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
            auto tensorWB = tla::MakeTensor(gmWB, params.layoutWB, Arch::PositionGM{});
            PaddingB paddingB(resource);
            paddingB(tensorWB, tensorB);
        }

        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementA> gmWA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        gmWA.SetGlobalBuffer((__gm__ ElementA*)params.ptrSymmetric);

        __gm__ int32_t* sigAddr = reinterpret_cast<__gm__ int32_t*>(params.ptrSignal);

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorSymmetric = tla::MakeTensor(gmWA, params.layoutWA, Arch::PositionGM{});

        // Reuse communication buffers and event IDs across transfer steps.
        AscendC::LocalTensor<ElementA> commTmpBuffer[COMM_BUFFER_NUM];
        uint64_t commBufferOffset = 0;
        for (uint32_t i = 0; i < COMM_BUFFER_NUM; ++i) {
            commTmpBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementA>(commBufferOffset);
            commBufferOffset += COMM_BUFFER_BYTES;
        }
        AscendC::TEventID commEventIds[COMM_BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
        uint32_t commBufferIndex = 0;

        uint32_t rankId = aclshmem_my_pe();
        uint32_t rankSize = aclshmem_n_pes();

        uint32_t flagOffset = 1;
        for (uint32_t cocIdx = 1; cocIdx < rankSize; cocIdx = cocIdx * 2) {
            // 本轮与 rankId ^ cocIdx 配对，交换两个已收集的数据分组。
            uint32_t remoteId = ((rankId + cocIdx) % (cocIdx * 2)) + (rankId / (cocIdx * 2)) * (cocIdx * 2);
            uint32_t nextCocIdx = cocIdx * 2;
            uint32_t nextRemoteId =
                ((rankId + nextCocIdx) % (nextCocIdx * 2)) + (rankId / (nextCocIdx * 2)) * (nextCocIdx * 2);

            bool isFirstIter = (cocIdx == 1);
            uint32_t offset = ((rankId / cocIdx) % 2) == 0 ? 1 : -1;
            uint32_t symmOffset = cocIdx * ((rankId / cocIdx) + offset);

            // 本 rank 的原始 A 数据。
            auto tensorBlockA = GetTile(
                tensorA, tla::MakeCoord(static_cast<uint32_t>(0), static_cast<uint32_t>(0)),
                tla::MakeShape(params.problemShape.m(), params.problemShape.k()));

            // 首轮使用本 rank 槽位，后续收集结果写入配对分组的槽位。
            auto tensorBlockSymmetricForFirst = GetTile(
                tensorSymmetric, tla::MakeCoord(rankId * params.problemShape.m(), static_cast<uint32_t>(0)),
                tla::MakeShape(params.problemShape.m() * cocIdx, params.problemShape.k()));

            auto tensorBlockSymmetric = GetTile(
                tensorSymmetric, tla::MakeCoord(symmOffset * params.problemShape.m(), static_cast<uint32_t>(0)),
                tla::MakeShape(params.problemShape.m() * cocIdx, params.problemShape.k()));

            if (cocIdx != 1) { // 等待远程rank将数据准备就绪,即上一轮完成
                aclshmem_signal_wait_until(sigAddr + ((flagOffset - 1) * 32), ACLSHMEM_CMP_EQ, 1);
            }

            AllGatherStep(
                tensorBlockSymmetric, tensorBlockSymmetricForFirst, tensorBlockA, sigAddr, rankId, rankSize, remoteId,
                isFirstIter, commTmpBuffer, commEventIds, commBufferIndex);

            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            if ((AscendC::GetBlockIdx() == 0) && (cocIdx * 2 < rankSize)) { // 通知下一轮的远程rank，数据准备就绪
                aclshmemx_signal_op(sigAddr + (flagOffset * 32), 1, ACLSHMEM_SIGNAL_SET, nextRemoteId);
            }
            flagOffset += 1;
        }
        // 矩阵A数据准备就绪，开启mmad
        Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishAllGather);
    }

private:
    static constexpr uint32_t COMM_BUFFER_NUM = 2;
    static constexpr uint32_t COMM_BUFFER_BYTES = 96 * 1024;
    static constexpr uint32_t COMM_COMPUTE_LENGTH = COMM_BUFFER_BYTES / sizeof(ElementA);

    using LayoutInner = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
    using TensorInnerUb = tla::Tensor<
        AscendC::LocalTensor<ElementA>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;
    using TensorInnerSrcGm =
        tla::Tensor<AscendC::GlobalTensor<ElementA>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorInnerDstGm =
        tla::Tensor<AscendC::GlobalTensor<ElementA>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    using CopyGm2Ub = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorInnerSrcGm, TensorInnerUb>;

    using RemoteCopyType = Catlass::Gemm::GemmType<ElementA, Catlass::layout::RowMajor>;
    using TileRemoteCopy = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, true, RemoteCopyType, RemoteCopyType, void, Catccos::detail::CopyDirect::Get,
        Catccos::detail::CopyTransport::Mte>;
    using TileRemoteCopyLocalUbToShmem =
        Catccos::Comm::Tile::TileRemoteCopyLocalUbToShmem<ArchTag, TensorInnerDstGm, TensorInnerUb>;

    template <class Tensor>
    CATLASS_DEVICE auto GetPaddingTensor(Tensor const& tensor)
    {
        if constexpr (std::is_same_v<typename Tensor::Layout, LayoutInner>) {
            return tensor;
        } else {
            auto shape = tla::MakeShape(tla::get<1>(tensor.shape()), tla::get<0>(tensor.shape()));
            auto stride = tla::MakeStride(tla::get<1>(tensor.stride()), tla::get<0>(tensor.stride()));
            return tla::MakeTensor(tensor.data(), tla::MakeLayout(shape, stride), Arch::PositionGM{});
        }
    }

    template <class TensorOut, class TensorIn>
    CATLASS_DEVICE void AllGatherStep(
        TensorOut& tensorSymm, TensorOut& tensorSymmForFirst, TensorIn const& tensorA, __gm__ int32_t* sigAddr,
        uint32_t rankId, uint32_t rankSize, uint32_t remoteId, bool isFirstIter,
        AscendC::LocalTensor<ElementA> (&tmpBuffer)[COMM_BUFFER_NUM],
        AscendC::TEventID const (&eventIds)[COMM_BUFFER_NUM], uint32_t& bufferIndex)
    {
        CopyGm2Ub copyGm2Ub;
        TileRemoteCopy tileRemoteCopy;
        TileRemoteCopyLocalUbToShmem copyLocalUbToShmem;
        // 行优先保持原本数据布局，列优先交换两维
        auto paddingTensorSrc = GetPaddingTensor(tensorA);
        auto paddingTensorDst = GetPaddingTensor(tensorSymm);

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t aivSubId = AscendC::GetSubBlockIdx();

        uint32_t tilesNum = tla::get<0>(paddingTensorDst.shape());
        uint32_t tileLen = tla::get<1>(paddingTensorDst.shape());
        uint32_t roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(ElementA)>(tileLen); // 32B对齐

        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv;
        if (aivId >= tileRemain) {
            mIdx += tileRemain;
        }
        MatrixCoord blockOffset(mIdx, 0);

        uint32_t coreLoops{0};
        uint32_t tilesPerLoop = COMM_COMPUTE_LENGTH / roundTileLen;
        coreLoops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;

        if (isFirstIter) { // 首轮将本 rank 的 A 拷入本地对称内存的对应槽位。
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) {
                    actualTilesNum = tilesPerAiv - tileIdx;
                }
                auto offset = tla::MakeCoord(mIdx + tileIdx, static_cast<uint32_t>(0));

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                auto tensorTileA = GetTile(paddingTensorSrc, offset, tla::MakeShape(actualTilesNum, tileLen));

                auto layoutUb = tla::MakeLayout(
                    tla::MakeShape(actualTilesNum, tileLen),
                    tla::MakeStride(static_cast<int64_t>(roundTileLen), tla::Int<1>{}));
                auto tensorUb = tla::MakeTensor(tmpBuffer[bufferIndex], layoutUb, Arch::PositionUB{});

                copyGm2Ub(tensorUb, tensorTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                auto tensorTileSymmDst = GetTile(tensorSymmForFirst, offset, tla::MakeShape(actualTilesNum, tileLen));

                copyLocalUbToShmem(tensorTileSymmDst, tensorUb, rankId, eventIds[bufferIndex]);

                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % COMM_BUFFER_NUM;
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);

            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            if (aivId == 0) { // 通知配对 rank：本地 A 数据已经写入对称内存。
                aclshmemx_signal_op(sigAddr, 1, ACLSHMEM_SIGNAL_SET, remoteId);
            }

            // 首轮先等待配对 rank 将其 A 数据写入对应的对称内存。
            aclshmem_signal_wait_until(sigAddr, ACLSHMEM_CMP_EQ, 1);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
        for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
            uint32_t tileIdx = loopIdx * tilesPerLoop;
            uint32_t actualTilesNum = tilesPerLoop;
            if (tilesPerAiv - tileIdx < tilesPerLoop) {
                actualTilesNum = tilesPerAiv - tileIdx;
            }
            auto offset = tla::MakeCoord(mIdx + tileIdx, static_cast<uint32_t>(0));

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
            auto tensorTileSymmSrc = GetTile(paddingTensorDst, offset, tla::MakeShape(actualTilesNum, tileLen));

            auto tensorTileSymmDst = GetTile(paddingTensorDst, offset, tla::MakeShape(actualTilesNum, tileLen));
            MatrixCoord copyShape{tla::get<0>(tensorTileSymmSrc.shape()), tla::get<1>(tensorTileSymmSrc.shape())};
            Catlass::layout::RowMajor srcLayout{
                copyShape.row(), copyShape.column(), tla::get<0>(tensorTileSymmSrc.stride())};
            Catlass::layout::RowMajor dstLayout{
                copyShape.row(), copyShape.column(), tla::get<0>(tensorTileSymmDst.stride())};
            auto srcOffset = tensorTileSymmSrc.layout()(tensorTileSymmSrc.coord());
            auto dstOffset = tensorTileSymmDst.layout()(tensorTileSymmDst.coord());
            tileRemoteCopy(
                tensorTileSymmDst.data()[dstOffset], dstLayout, tensorTileSymmSrc.data()[srcOffset], srcLayout,
                copyShape, tmpBuffer[bufferIndex], eventIds[bufferIndex], remoteId);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

            bufferIndex = (bufferIndex + 1) % COMM_BUFFER_NUM;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    }

    static constexpr Arch::FlagID FLAG_AIV_FINISH_ALLGATHER = 0;
    static constexpr Arch::FlagID RV_FLAG_AIV_FINISH_ALLGATHER = 1;
    Arch::CrossCoreFlagWithReverse<> flagAivFinishAllGather{FLAG_AIV_FINISH_ALLGATHER, RV_FLAG_AIV_FINISH_ALLGATHER};

    Arch::Resource<ArchTag> resource;
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_ALLGATHER_MATMUL_SHALLOW_FUSION_HPP
