#pragma once

#include <tuple>
#include <vector>
#include <thread>
#include "auction_heap.h"

namespace auction
{
  template <class I, class SC>
  class FindLinear
  {
  private:
    int target_size;
  public:
    FindLinear(int target_size) : target_size(target_size) {}

    bool findBid(I &iterator, int x, std::pair<int, int> &y, std::pair<SC, SC> &cost, std::vector<SC> &beta)
    {
      y.first = y.second = -1;
      cost.first = cost.second = MAX_COST;
      // y.1 = argmin(...) and y.2 = min(...) with y.2 in Y\y.1
      iterator.iterate([&](int yy, SC ccost)
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
    void fillCache(I &iterator, int x, std::pair<int, int> &y, std::pair<SC, SC> &cost, std::vector<SC> &beta) {}
    // nothing here
    void fixBeta(SC dlt) {}
  };
}
