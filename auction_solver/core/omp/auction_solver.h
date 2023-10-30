#pragma once
#pragma once

#include <chrono>
#include <sstream>
#include <iostream>
#include <cstring>
#ifndef LAP_QUIET
#include <deque>
#include <mutex>
#endif
#include <math.h>

namespace auction
{
  namespace omp
  {
    template <class SC, class I>
    SC guessEpsilon(int dim, int dim2, I& iterator)
    {
      SC epsilon(0);
#pragma omp parallel for reduction(+:epsilon)
      for (int x = 0; x < dim; x++)
      {
        auto tt = iterator.getRow(x);
        SC min_cost, max_cost;
        min_cost = max_cost = tt[0];
        for (int y = 1; y < dim2; y++)
        {
          SC cost_l = (SC)tt[y];
          min_cost = std::min(min_cost, cost_l);
          max_cost = std::max(max_cost, cost_l);
        }
        epsilon += max_cost - min_cost;
      }
      return (epsilon / SC(10 * dim));
    }

    template <class SC, class CF, class I, class F>
    void solve(int dim, int dim2, CF &costfunc, I &iterator, F &find, std::vector<SC> &beta, int *rowsol, bool use_epsilon)

      // input:
      // dim/dim2   - problem size
      // costfunc   - cost matrix
      // iterator   - iterator for accessing the cost matrix

      // optional input (defaults to tue)
      // used_epsilon - if true, epsilon scaling will be used
      // find_caching - if true, caching of last sqrt(dim) best values is used

      // output:
      // rowsol     - column assigned to row in solution

    {
#ifdef DISPLAY_MISS
      long long hit_count = 0;
      long long miss_count = 0;

      long long total_hit_count = 0;
      long long total_miss_count = 0;
#endif
#ifndef LAP_QUIET
      auto start_time = std::chrono::high_resolution_clock::now();
#endif

      std::vector<int> coupling(dim2);
      int target_size = (int)coupling.size();
      std::vector<std::pair<volatile int, volatile SC>> B(target_size);
      std::vector<std::vector<int>> unassigned(2);
      unassigned[0].resize(target_size);
      unassigned[1].resize(target_size);

      std::vector<int> receiving(target_size);
      int r_count = 0;

      std::vector<int> u_count(2);
      u_count[0] = u_count[1] = 0;

#ifdef DEFER_MISS
      std::vector<int> deferred(target_size);
      int d_count = 0;
#endif

      SC epsilon = auction::omp::guessEpsilon<SC>(dim, dim2, iterator);

      int completed = 0;

#pragma omp parallel for
      for (int y = 0; y < target_size; y++)
      {
        B[y].first = -1;
        B[y].second = MAX_COST;
        beta[y] = SC(0);
      }

      std::vector<std::mutex> Block(B.size());
      // loop until done

#ifndef LAP_QUIET
      int elapsed = 0;
#endif
      SC epsilon_lower = epsilon / SC(1000 * target_size);
      int iteration = 0;

#ifdef LAP_QUIET
      auction::omp::AuctionOneWay<SC> auction;
#else
      auction::omp::AuctionOneWay<SC> auction(hit_count, miss_count, total_hit_count, total_miss_count);
#endif

      bool first = true;

      SC max_beta(0);
 
      while (epsilon >= SC(0))
      {
        completed = 0;

        if (epsilon > SC(0))
        {
          if (!first)
          {
            epsilon *= SC(0.2);
            if (epsilon < epsilon_lower) epsilon = SC(0);
          }
        }
        first = false;

        // initialize coupling
        for (int i = 0; i < target_size; i++)
        {
          coupling[i] = -1;
          unassigned[0][i] = i;
        }
        u_count[0] = target_size;

#ifndef LAP_QUIET
        std::stringstream ss;
        ss << " eps = " << epsilon;
        lap::displayTime(start_time, ss.str().c_str(), std::cout);
#endif
        while (completed < target_size)
        {
#ifndef LAP_QUIET
          std::stringstream ss;
          ss << " eps = " << epsilon;
          lap::displayProgress(start_time, elapsed, completed, target_size, ss.str().c_str(), iteration);
#endif
          iteration++;

          if (epsilon != 0.0f)
          {
#ifdef DEFER_MISS
            completed = auction.template auction<false>(coupling, beta, iterator, find, u_count, unassigned, d_count, deferred, r_count, receiving, B, Block, target_size, epsilon);
#else
            completed = auction.template auction<false>(coupling, beta, iterator, find, u_count, unassigned, r_count, receiving, B, Block, target_size, epsilon);
#endif
          }
          else
          {
#ifdef DEFER_MISS
            completed = auction.template auction<true>(coupling, beta, iterator, find, u_count, unassigned, d_count, deferred, r_count, receiving, B, Block, target_size, epsilon);
#else
            completed = auction.template auction<true>(coupling, beta, iterator, find, u_count, unassigned, r_count, receiving, B, Block, target_size, epsilon);
#endif
          }

#ifndef LAP_QUIET
          if (completed == target_size)
          {
#pragma omp parallel for
            for (int x = 0; x < N; x++)
            {
              if (coupling[x] != -1) rowsol[coupling[x]] = x;
            }
            AC ccost = this.cost(dim, dim2, costfunc, rowsol);
            std::stringstream ss;
            ss << " epsilon = " << epsilon;
            if (ccost > SC(0)) ss << " cost = " << ccost;
            auction::displayProgress(start_time, elapsed, completed, target_size, ss.str().c_str(), iteration);
          }
#endif
        }

        // check for epsilon == 0 and ensure termination
        if (epsilon == SC(0)) epsilon = SC(-1);
        else
        {
          // find maximum beta (for stability)
          max_beta = beta[0];
#pragma omp parallel for reduction(max:max_beta)
          for (int y = 1; y < target_size; y++) max_beta = std::max(max_beta, beta[y]);
#pragma omp parallel for
          for (int y = 0; y < target_size; y++) beta[y] -= max_beta;
          find.fixBeta(max_beta);
        }
      }
#pragma omp parallel for
      for (int x = 0; x < dim; x++)
      {
        if (coupling[x] != -1) rowsol[coupling[x]] = x;
      }
    }

    template <class SC, class CF, class I>
    void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon, bool find_caching)

      // input:
      // dim/dim2   - problem size
      // costfunc   - cost matrix
      // iterator   - iterator for accessing the cost matrix

      // optional input (defaults to tue)
      // used_epsilon - if true, epsilon scaling will be used
      // find_caching - if true, caching of last sqrt(dim) best values is used

      // output:
      // rowsol     - column assigned to row in solution

    {
      std::vector<SC> beta(dim);
      int cache_size = 3;
      if (find_caching)
      {
        FindCaching<I, SC> find(dim, iterator, beta, cache_size);
        auction::omp::solve(dim, dim2, costfunc, iterator, find, beta, rowsol, use_epsilon);
      } else {
        FindLinear<I, SC> find(dim);
        auction::omp::solve(dim, dim2, costfunc, iterator, find, beta, rowsol, use_epsilon);
      }
    }

    // shortcut for square problems
    template <class SC, class CF, class I>
    void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon, bool find_caching)
    {
      auction::omp::solve<SC>(dim, dim, costfunc, iterator, rowsol, use_epsilon, find_caching);
    }

    template <class SC, class CF>
    SC cost(int dim, int dim2, CF &costfunc, int *rowsol)
    {
      SC total = SC(0);
#pragma omp parallel for reduction(+:total)
      for (int i = 0; i < dim; i++) total += costfunc.getCost(i, rowsol[i]);
      return total;
    }

    template <class SC, class CF>
    SC cost(int dim, CF &costfunc, int *rowsol)
    {
      return auction::omp::cost<SC, CF>(dim, dim, costfunc, rowsol);
    }
  }
}
