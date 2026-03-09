
#ifndef CATCCOS_DETAIL_REMOTE_COPY_TYPE_HPP
#define CATCCOS_DETAIL_REMOTE_COPY_TYPE_HPP

namespace Catccos::detail {

enum class CopyDirect {Put, Get};
enum class CopyTransport {Mte, Rdma};

} // namespace Catccos::detail

#endif // CATCCOS_DETAIL_REMOTE_COPY_TYPE_HPP