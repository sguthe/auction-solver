#pragma once

#include <algorithm>
#include <vector>

namespace auction
{
  template <class TC, class CF>
  class DirectIterator
  {
  public:
    CF &costfunc;
  public:
    DirectIterator(CF &costfunc) : costfunc(costfunc) {}
    ~DirectIterator() {}

    void getHitMiss(long long &hit, long long &miss) { hit = miss = 0; }

    __forceinline const TC *getRow(int i) { return costfunc.getRow(i); }
   // __forceinline const TC getCost(int i, int j) { return costfunc.getCost(i, j); }

    template <class C>
    void iterate(const C &c, int x, int size_y, TC &limit, std::vector<TC> &beta)
    {
      const TC *row = getRow(x);
      for (int yy = 0; yy < size_y; yy++)
      {
        if (-beta[yy] <= limit)
        {
          TC ccost = row[yy] - beta[yy];
          c(yy, ccost);
        }
      }
    }

    template <class C>
    void iterate(const C &c, int x, int size_y, std::vector<TC> &beta)
    {
      const TC *row = getRow(x);
      for (int yy = 0; yy < size_y; yy++)
      {
        TC ccost = row[yy] - beta[yy];
        c(yy, ccost);
      }
    }
  };
}

