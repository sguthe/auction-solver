#pragma once

#if defined(__GNUC__)
#define __forceinline \
        __inline__ __attribute__((always_inline))
#endif

#ifdef LAP_QUIET
#define lapAssert(A)
#else
// note: this will automatically be disabled if NDEBUG is set
#include <assert.h>
#define lapAssert(A) assert(A)
#endif

#ifndef lapInfo
#define lapInfo std::cout
#endif

#ifndef lapDebug
#define lapDebug std::cout
#endif

#ifndef lapAlloc
#define lapAlloc auction::alloc
#endif

#ifndef lapFree
#define lapFree auction::free
#endif

namespace auction
{
  // Functions used for solving the lap, calculating the costs of a certain assignment and guessing the initial epsilon value.
  template <class SC, class CF, class I> void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon = true, bool find_caching = true);
  template <class SC, class CF, class I> void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon = true, bool find_caching = true);
  template <class SC, class CF> SC cost(int dim, CF &costfunc, int *rowsol);
  template <class SC, class CF> SC cost(int dim, int dim2, CF &costfunc, int *rowsol);

  // Cost functions, including tabulated costs
  template <class TC, typename GETCOST> class SimpleCostFunction;
  template <class TC> class TableCost;

  // Iterator classes used for accessing the cost functions
  template <class TC, class CF> class DirectIterator;

  // Memory management
  template <typename T> void alloc(T * &ptr, unsigned long long width, const char *file, const int line);
  template <typename T> void free(T *&ptr);

#ifdef LAP_OPENMP
  namespace omp
  {
    // Functions used for solving the lap, calculating the costs of a certain assignment and guessing the initial epsilon value.
    template <class SC, class CF, class I> void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon = true, bool find_caching = true);
    template <class SC, class CF, class I> void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon = true, bool find_caching = true);
    template <class SC, class CF> SC cost(int dim, CF &costfunc, int *rowsol);
    template <class SC, class CF> SC cost(int dim, int dim2, CF &costfunc, int *rowsol);

    // Cost functions, including tabulated costs
    template <class TC, typename GETCOST> class SimpleCostFunction;
    template <class TC> class TableCost;

    // Iterator classes used for accessing the cost functions
    template <class TC, class CF> class DirectIterator;

  }
#endif

}

#include "core/auction_cost.h"
#include "core/auction_direct_iterator.h"
#include "core/auction_find_caching.h"
#include "core/auction_find_linear.h"
#include "core/auction_heap.h"
#include "core/auction_helper.h"
#include "core/auction_one_way.h"
#include "core/auction_solver.h"
#ifdef LAP_OPENMP
#include "core/omp/auction_direct_iterator.h"
#include "core/omp/auction_find_caching.h"
#include "core/omp/auction_find_linear.h"
#include "core/omp/auction_one_way.h"
#include "core/omp/auction_solver.h"
#endif
