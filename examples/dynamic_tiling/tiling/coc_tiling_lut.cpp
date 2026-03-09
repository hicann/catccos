#include "coc_tiling_lut.h"
#include <iostream>
#include <limits>

static int GetValueFromMap(int64_t m, int64_t k, int64_t n,
  const std::map<int, std::vector<std::vector<int>>>& condMap,
  int defaultVal)
{
  for (const auto& [candidate, condList] : condMap) {
    for (const auto& c : condList) {
      auto in = [&](int64_t v, int lo, int hi) {
        return (lo == -1 || v >= lo) && (hi == -1 || v <= hi);
      };
      if (in(m, c[0], c[1]) && in(k, c[2], c[3]) && in(n, c[4], c[5]))
        return candidate;
    }
  }
  return defaultVal;
}

const std::map<LutKey, const LUTGroup*> g_allLutGroups = {
  {{MATMUL_ALLREDUCE, 2}, &AllReduce2p},
  {{MATMUL_ALLREDUCE, 4}, &AllReduce4p},
  {{MATMUL_ALLREDUCE, 8}, &AllReduce8p},
  {{MATMUL_REDUCE_SCATTER, 2}, &ReduceScatter2p},
  {{MATMUL_REDUCE_SCATTER, 4}, &ReduceScatter4p},
  {{MATMUL_REDUCE_SCATTER, 8}, &ReduceScatter8p},
  {{ALLGATHER_MATMUL, 2}, &AllGather2p},
  {{ALLGATHER_MATMUL, 4}, &AllGather4p},
  {{ALLGATHER_MATMUL, 8}, &AllGather8p},
  {{ALLGATHER_MATMUL_WITH_GATHER_RESULT, 2}, &AllGather2p},
  {{ALLGATHER_MATMUL_WITH_GATHER_RESULT, 4}, &AllGather4p},
  {{ALLGATHER_MATMUL_WITH_GATHER_RESULT, 8}, &AllGather8p},
  {{ALLGATHER_MATMUL_RDMA, 2}, &AllGather2p},
  {{ALLGATHER_MATMUL_RDMA, 4}, &AllGather4p},
  {{ALLGATHER_MATMUL_RDMA, 8}, &AllGather8p},
  // 继续添加...
};

bool ApplyLookupTable(const COCMatMulInfo& info,
                     CocCommType type,
                     int rankSize,
                     CocTilingParams& t)
{
  LutKey key = {type, rankSize};
  auto it = g_allLutGroups.find(key);
  if (it == g_allLutGroups.end()) {
    std::cerr << "[LUT] no table for (" << type << ',' << rankSize << ")\n";
    return false;
  }
  const LUTGroup& g = *(it->second); // 解引用指针
  auto pick = [&](auto& mp, int def) { return GetValueFromMap(info.m, info.k, info.n, mp, def); };
  t.m0 = pick(g.m0Map, g.m0Default);
  t.commInterval = pick(g.commIntervalMap, g.commIntervalDefault);
  t.commTileM = pick(g.commTileMMap, g.commTileMDefault) * 2;
  t.commNpuSplit = pick(g.commNpuSplitMap, g.commNpuSplitDefault);
  t.commDataSplit = pick(g.commDataSplitMap, g.commDataSplitDefault);
  t.commBlockM = t.commTileM;
  t.n0 = (t.m0 == 256) ? 128 : 256;
  t.k0 = 256;
  return true;
}