#pragma once

#include "auction_helper.h"

#ifdef LAP_OPENMP
#  include <omp.h>
#endif

#include <vector>

namespace auction
{
  template <class AC, class IT>
  class AdaptorCost
  {
  private:
    int target_size;
    IT &iterator;

    std::vector<AC> *p_beta;
    std::vector<AC> *p_alpha;

  public:
    AdaptorCost(IT &iterator, int target_size)
      : target_size(target_size), iterator(iterator)
    {
    }

    ~AdaptorCost()
    {
    }

    AC getCost(int x, int y) { return iterator.getRow(x)[y]; }

    template <bool PAR, class C>
    void iterate(const C &c, int x, AC &limit, std::vector<AC> &beta)
    {
      const AC *row = iterator.getRow(x);
      if (PAR)
      {
#pragma omp parallel for
        for (int yy = 0; yy < target_size; yy++)
        {
          if (-beta[yy] <= limit)
          {
            AC ccost = row[yy] - beta[yy];
            c(yy, ccost);
          }
        }
      }
      else
      {
        for (int yy = 0; yy < target_size; yy++)
        {
          if (-beta[yy] <= limit)
          {
            AC ccost = row[yy] - beta[yy];
            c(yy, ccost);
          }
        }
      }
    }

    template <bool PAR, class C>
    void iterate(const C &c, int x, std::vector<AC> &beta)
    {
      AC *row = iterator.getRow(x);
      if (PAR)
      {
#pragma omp parallel for
        for (int yy = 0; yy < target_size; yy++)
        {
          AC ccost = row[yy] - beta[yy];
          c(yy, ccost);
        }
      }
      else
      {
        for (int yy = 0; yy < target_size; yy++)
        {
          AC ccost = row[yy] - beta[yy];
          c(yy, ccost);
        }
      }
    }

    void updateBeta(std::vector<AC> &beta, int r_count) {
      p_beta = &beta;
    }
    void updateAlpha(std::vector<AC> &alpha, int r_count) {
      p_alpha = &alpha;
    }

    void printCases(bool reset) {}

    bool requiresCaching() { return true; }
  };
}
