#include "allgather_matmul_dequant_bias/allgather_matmul_dequant_bias_device.h"
#include "allgather_matmul_dequant/allgather_matmul_dequant_device.h"
#include "allgather_matmul_dequant/allgather_matmul_dequant_padding_device.h"

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = half;
using ElementScale = uint64_t;

using LayoutA0 = Catlass::layout::RowMajor;
using LayoutB0 = Catlass::layout::RowMajor;

using LayoutA1 = Catlass::layout::ColumnMajor;
using LayoutB1 = Catlass::layout::ColumnMajor;

using LayoutC = Catlass::layout::RowMajor;
using LayoutScale = Catlass::layout::VectorLayout;

void LaunchAllGatherMatmulDequantBiasINT8(
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
    uint8_t *biasPtr = kernelParams.customPtrs[0];
    uint8_t *scalePtr = kernelParams.customPtrs[1];
    AllGatherMatmulDequantBias<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC, ElementScale, LayoutScale>
        <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, cPtr, biasPtr, scalePtr, symmetricPtr, cocTiling);
}

void LaunchAllGatherMatmulDequantINT8(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *dPtr = kernelParams.ptrC;
    uint8_t *scalePtr = kernelParams.customPtrs[0];
    AllGatherMatmulDequant<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC, ElementScale, LayoutScale>
        <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, dPtr, scalePtr, workSpace, symmetricPtr, cocTiling);
}

void LaunchAllGatherMatmulDequantPaddingINT8(
    void *stream, uint64_t fftsAddr,
    KernelParams& kernelParams,
    uint8_t *workSpace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    uint8_t *aPtr = kernelParams.ptrA;
    uint8_t *bPtr = kernelParams.ptrB;
    uint8_t *dPtr = kernelParams.ptrC;
    uint8_t *scalePtr = kernelParams.customPtrs[0];
    AllGatherMatmulDequantPadding<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC, ElementScale, LayoutScale>
        <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, aPtr, bPtr, dPtr, scalePtr, workSpace, symmetricPtr, cocTiling);
}