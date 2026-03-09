#include "matmul_allreduce/matmul_allreduce_device.h"
#include "allgather_matmul/allgather_matmul_device.h"
#include "matmul_reduce_scatter/matmul_reduce_scatter_device.h"
#include "allgather_matmul_with_gather_result/allgather_matmul_with_gather_result_device.h"
#include "grouped_matmul_alltoallv/grouped_matmul_alltoallv_device.h"
#include "alltoallv_grouped_matmul/alltoallv_grouped_matmul_device.h"
#include "alltoallv_gmm_v2/alltoallv_gmm_v2_device.h"

#ifdef RDMA_TRANSPORT
#include "allgather_matmul_rdma/allgather_matmul_rdma_device.h"
#endif

using namespace AscendC;

using ElementA = bfloat16_t;
using ElementB = bfloat16_t;
using ElementC = bfloat16_t;

using LayoutA0 = Catlass::layout::RowMajor;
using LayoutB0 = Catlass::layout::RowMajor;

using LayoutA1 = Catlass::layout::ColumnMajor;
using LayoutB1 = Catlass::layout::ColumnMajor;

using LayoutC = Catlass::layout::RowMajor;

void LaunchMatmulAllReduceBF16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *cPtr = kernelParams.ptrC;
    if (!transA && !transB) {
        MatmulAllReduce<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        MatmulAllReduce<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, symmetricPtr, cocTiling);
    } else if (transA && !transB) {
        MatmulAllReduce<ElementA, LayoutA1, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, symmetricPtr, cocTiling);
    } else {
        MatmulAllReduce<ElementA, LayoutA1, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, symmetricPtr, cocTiling);
    }
}

void LaunchAllGatherMatmulBF16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *cPtr = kernelParams.ptrC;
    if (!transA && !transB) {
        AllGatherMatmul<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        AllGatherMatmul<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, symmetricPtr, cocTiling);
    }
}

void LaunchMatmulReduceScatterBF16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *cPtr = kernelParams.ptrC;
    if (!transA && !transB) {
        MatmulReduceScatter<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        MatmulReduceScatter<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, symmetricPtr, cocTiling);
    } else if (transA && !transB) {
        MatmulReduceScatter<ElementA, LayoutA1, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, symmetricPtr, cocTiling);
    } else {
        MatmulReduceScatter<ElementA, LayoutA1, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, symmetricPtr, cocTiling);
    }
}

void LaunchAllGatherMatmulWithGatherResultBF16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *cPtr = kernelParams.ptrC;
    uint8_t *gatherAPtr = kernelParams.customPtrs[0];
    if (!transA && !transB) {
        AllGatherMatmulWithGatherResult<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, gatherAPtr, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        AllGatherMatmulWithGatherResult<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, gatherAPtr, symmetricPtr, cocTiling);
    }
}

void LaunchGroupedMatmulAllToAllVBF16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *cPtr = kernelParams.ptrC;
    uint8_t *localExpertPtr = kernelParams.customPtrs[0];
    uint8_t *globalExpertPtr = kernelParams.customPtrs[1];
    if (!transA && !transB) {
        GroupedMatmulAllToAllV<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, localExpertPtr, globalExpertPtr, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        GroupedMatmulAllToAllV<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, localExpertPtr, globalExpertPtr, symmetricPtr, cocTiling);
    } else if (transA && !transB) {
        GroupedMatmulAllToAllV<ElementA, LayoutA1, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, localExpertPtr, globalExpertPtr, symmetricPtr, cocTiling);
    } else {
        GroupedMatmulAllToAllV<ElementA, LayoutA1, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, localExpertPtr, globalExpertPtr, symmetricPtr, cocTiling);
    }
}

void LaunchAllToAllVGroupedMatmulBF16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *cPtr = kernelParams.ptrC;
    uint8_t *localExpertPtr = kernelParams.customPtrs[0];
    uint8_t *globalExpertPtr = kernelParams.customPtrs[1];
    if (!transA && !transB) {
        AllToAllVGroupedMatmul<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, localExpertPtr, globalExpertPtr, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        AllToAllVGroupedMatmul<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, localExpertPtr, globalExpertPtr, symmetricPtr, cocTiling);
    }
}

#ifdef RDMA_TRANSPORT
void LaunchAllGatherMatmulRdmaBF16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workSpace;
    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *cPtr = kernelParams.ptrC;
    if (!transA && !transB) {
        AllGatherMatmulRdma<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        AllGatherMatmulRdma<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, symmetricPtr, cocTiling);
    }
}
#endif

void LaunchAllToAllVGMMV2BF16(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *cPtr = kernelParams.ptrC;
    uint8_t *tokenPerExpertPtr = kernelParams.customPtrs[0];
    if (!transA && !transB) {
        AllToAllVGMMV2<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, tokenPerExpertPtr, workSpace, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        AllToAllVGMMV2<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, tokenPerExpertPtr, workSpace, symmetricPtr, cocTiling);
    }
}