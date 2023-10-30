#pragma once

#include <tuple>
#include <vector>
#include <thread>
#include "auction_heap.h"

namespace auction
{
  template <class I, class AC>
  class FindLinear
  {
  private:
    int target_size;
  public:
    FindLinear(int target_size) : target_size(target_size) {}

    bool findBid(I &iterator, int x, std::pair<int, int> &y, std::pair<AC, AC> &cost, std::vector<AC> &beta)
    {
      y.first = y.second = -1;
      cost.first = cost.second = MAX_COST;
      // y.1 = argmin(...) and y.2 = min(...) with y.2 in Y\y.1
      iterator.template iterate([&](int yy, AC ccost)
      {
        if ((ccost < cost.first) || ((ccost == cost.first) && (yy < y.first)))
        {
          y.second = y.first;
          y.first = yy;
          cost.second = cost.first;
          cost.first = ccost;
        }
        else if ((ccost < cost.second) || ((ccost == cost.second) && (yy < y.second)))
        {
          y.second = yy;
          cost.second = ccost;
        }
      }, x, target_size, cost.second, beta);
      return true;
    }
    // nothing here
    template <bool FILL_ONLY>
    void fillCache(I &iterator, int x, std::pair<int, int> &y, std::pair<AC, AC> &cost, std::vector<AC> &beta) {}
    // nothing here
    void fixBeta(AC dlt) {}
  };
}
