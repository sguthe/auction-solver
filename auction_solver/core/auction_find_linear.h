#pragma once

#include <tuple>
#include <vector>
#include <thread>
#ifdef LAP_OPENMP
#  include <omp.h>
#endif
#include "auction_heap.h"

namespace auction
{
  template <class COST, class AC>
  class FindLinear
  {
  private:
    int target_size;
  public:
    FindLinear(int target_size) : target_size(target_size) {}

    template <bool PAR>
    bool findBid(COST &c, int x, std::pair<int, int> &y, std::pair<AC, AC> &cost, std::vector<AC> &beta)
    {
      y.first = y.second = -1;
      cost.first = cost.second = MAX_COST;
      // y.1 = argmin(...) and y.2 = min(...) with y.2 in Y\y.1
      c.template iterate<PAR>([&](int yy, AC ccost)
      {
        if (PAR)
        {
          if (ccost <= cost.second)
          {
#pragma omp critical
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
            }
          }
        }
        else
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
        }
      }, x, cost.second, beta);
      return true;
    }
    // nothing here
    template <bool PAR, bool FILL_ONLY>
    void fillCache(COST &c, int x, std::pair<int, int> &y, std::pair<AC, AC> &cost, std::vector<AC> &beta) {}
    // nothing here
    void fixBeta(AC dlt) {}
  };
}
