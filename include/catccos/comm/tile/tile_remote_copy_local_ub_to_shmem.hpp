#ifndef CATCCOS_COMM_TILE_REMOTE_COPY_LOCAL_UB_TO_SHMEM_HPP
#define CATCCOS_COMM_TILE_REMOTE_COPY_LOCAL_UB_TO_SHMEM_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "shmem.h"
#include "tla/tensor.hpp"

namespace Catccos::Comm::Tile {
using namespace Catlass;

template <class ArchTag, class TensorOut_, class TensorIn_>
class TileRemoteCopyLocalUbToShmem {
public:
    using TensorOut = TensorOut_;
    using TensorIn = TensorIn_;

    CATLASS_DEVICE
    TileRemoteCopyLocalUbToShmem() {};

    template <class TensorOut, class TensorIn>
    CATLASS_DEVICE void operator()(TensorOut& tensorOut, TensorIn& tensorIn, uint32_t rankId, AscendC::TEventID eventId)
    {
        non_contiguous_copy_param copyParams;
        copyParams.repeat = tla::get<0>(tensorIn.shape());
        copyParams.length = tla::get<1>(tensorIn.shape());
        copyParams.src_ld = tla::get<0>(tensorIn.stride()); // Source row stride, in elements.
        copyParams.dst_ld = tla::get<0>(tensorOut.stride());

        auto dstOffset = tensorOut.layout()(tensorOut.coord());
        auto srcOffset = tensorIn.layout()(tensorIn.coord());

        aclshmemx_mte_put_nbi(tensorOut.data()[dstOffset], tensorIn.data()[srcOffset], copyParams, rankId, eventId);
    }
};

} // namespace Catccos::Comm::Tile

#endif // CATCCOS_COMM_TILE_REMOTE_COPY_LOCAL_UB_TO_SHMEM_HPP
