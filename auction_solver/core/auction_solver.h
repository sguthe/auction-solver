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
#ifndef LAP_QUIET
  class AllocationLogger
  {
    std::vector<std::deque<void *>> allocated;
    std::vector<std::deque<unsigned long long>> size;
    std::vector<std::deque<char *>> alloc_file;
    std::vector<std::deque<int>> alloc_line;
    std::vector<unsigned long long> peak;
    std::vector<unsigned long long> current;
    std::vector<std::string> name;
    std::mutex lock;
  private:
    std::string commify(unsigned long long n)
    {
      std::string s;
      int cnt = 0;
      do
      {
        s.insert(0, 1, char('0' + n % 10));
        n /= 10;
        if (++cnt == 3 && n)
        {
          s.insert(0, 1, ',');
          cnt = 0;
        }
      } while (n);
      return s;
    }

  public:
    AllocationLogger()
    {
      allocated.resize(1);
      size.resize(1);
      alloc_file.resize(1);
      alloc_line.resize(1);
      peak.resize(1);
      current.resize(1);
      name.resize(1);
      peak[0] = 0ull;
      current[0] = 0ull;
      name[0] = std::string("memory");
    }

    ~AllocationLogger() {}
    void destroy()
    {
      for (size_t i = 0; i < peak.size(); i++)
      {
        if (!name[i].empty())
        {
          lapInfo << "Peak " << name[i] << " usage:" << commify(peak[i]) << " bytes" << std::endl;
          if (allocated[i].empty()) continue;
          lapInfo << (char)toupper(name[i][0]) << name[i].substr(1) << " leak list:" << std::endl;
          while (!allocated[i].empty())
          {
            lapInfo << "  leaked " << commify(size[i].front()) << " bytes at " << std::hex << allocated[i].front() << std::dec << ": " << alloc_file[i].front() << ":" << alloc_line[i].front() << std::endl;
            size[i].pop_front();
            allocated[i].pop_front();
            alloc_file[i].pop_front();
            alloc_line[i].pop_front();
          }
        }
      }
    }

    template <class T>
    void free(int idx, T a)
    {
      std::lock_guard<std::mutex> guard(lock);
#ifdef LAP_DEBUG
#ifndef LAP_NO_MEM_DEBUG
      lapDebug << "Freeing memory at " << std::hex << (size_t)a << std::dec << std::endl;
#endif
#endif
      for (unsigned long long i = 0; i < allocated[idx].size(); i++)
      {
        if ((void *)a == allocated[idx][i])
        {
          current[idx] -= size[idx][i];
          allocated[idx][i] = allocated[idx].back();
          allocated[idx].pop_back();
          size[idx][i] = size[idx].back();
          size[idx].pop_back();
          alloc_line[idx][i] = alloc_line[idx].back();
          alloc_line[idx].pop_back();
          alloc_file[idx][i] = alloc_file[idx].back();
          alloc_file[idx].pop_back();
          return;
        }
      }
    }

    template <class T>
    void alloc(int idx, T *a, unsigned long long s, const char *file, const int line)
    {
      std::lock_guard<std::mutex> guard(lock);
#ifdef LAP_DEBUG
#ifndef LAP_NO_MEM_DEBUG
      lapDebug << "Allocating " << s * sizeof(T) << " bytes at " << std::hex << (size_t)a << std::dec << " \"" << file << ":" << line << std::endl;
#endif
#endif
      current[idx] += s * sizeof(T);
      peak[idx] = std::max(peak[idx], current[idx]);
      allocated[idx].push_back((void *)a);
      size[idx].push_back(s * sizeof(T));
      alloc_file[idx].push_back((char *)file);
      alloc_line[idx].push_back(line);
    }
  };

  static AllocationLogger allocationLogger;
#endif

  template <typename T>
  void alloc(T * &ptr, unsigned long long width, const char *file, const int line)
  {
    ptr = (T*)malloc(sizeof(T) * (size_t) width); // this one is allowed
#ifndef LAP_QUIET
    allocationLogger.alloc(0, ptr, width, file, line);
#endif
  }

  template <typename T>
  void free(T *&ptr)
  {
    if (ptr == (T *)NULL) return;
#ifndef LAP_QUIET
    allocationLogger.free(0, ptr);
#endif
    ::free(ptr); // this one is allowed
    ptr = (T *)NULL;
  }

  std::string getTimeString(long long ms)
  {
    char time[256];
    long long sec = ms / 1000;
    ms -= sec * 1000;
    long long min = sec / 60;
    sec -= min * 60;
    long long hrs = min / 60;
    min -= hrs * 60;
#if defined (_MSC_VER)
    sprintf_s(time, "%3d:%02d:%02d.%03d", (int)hrs, (int)min, (int)sec, (int)ms);
#else
    sprintf(time, "%3d:%02d:%02d.%03d", (int)hrs, (int)min, (int)sec, (int)ms);
#endif

    return std::string(time);
  }

  std::string getSecondString(long long ms)
  {
    char time[256];
    long long sec = ms / 1000;
    ms -= sec * 1000;
#if defined (_MSC_VER)
    sprintf_s(time, "%d.%03d", (int)sec, (int)ms);
#else
    sprintf(time, "%d.%03d", (int)sec, (int)ms);
#endif

    return std::string(time);
  }

  template <class TP, class OS>
  void displayTime(TP &start_time, const char *msg, OS &lapStream)
  {
    auto end_time = std::chrono::high_resolution_clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    lapStream << getTimeString(ms) << ": " << msg << " (" << getSecondString(ms) << "s)" << std::endl;
  }

  template <class TP>
  int displayProgress(TP &start_time, int &elapsed, int completed, int target_size, const char *msg = 0, int iteration = -1, bool display = false)
  {
    if (completed == target_size) display = true;

#ifndef LAP_DEBUG
    if (!display) return 0;
#endif

    auto end_time = std::chrono::high_resolution_clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

#ifdef LAP_DEBUG
    if ((!display) && (elapsed * 10000 < ms))
    {
      elapsed = (int)((ms + 10000ll) / 10000ll);
      lapDebug << getTimeString(ms) << ": solving " << completed << "/" << target_size;
      if (iteration >= 0) lapDebug << " iteration = " << iteration;
      if (msg != 0) lapDebug << msg;
      lapDebug << std::endl;
      return 2;
    }

    if (display)
#endif
    {
      elapsed = (int)((ms + 10000ll) / 10000ll);
      lapInfo << getTimeString(ms) << ": solving " << completed << "/" << target_size;
      if (iteration >= 0) lapInfo << " iteration = " << iteration;
      if (msg != 0) lapInfo << msg;
      lapInfo << std::endl;
      return 1;
    }
#ifdef LAP_DEBUG
    return 0;
#endif
  }

  template <class SC, class I>
  SC guessEpsilon(int dim, int dim2, I& iterator)
  {
    SC epsilon(0);
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
    std::vector<std::pair<int, SC>> B(target_size);
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

    SC epsilon = guessEpsilon<SC>(dim, dim2, iterator);

    int completed = 0;

    for (int y = 0; y < target_size; y++)
    {
      B[y].first = -1;
      B[y].second = MAX_COST;
      beta[y] = SC(0);
    }

    // loop until done

#ifndef LAP_QUIET
    int elapsed = 0;
#endif
    SC epsilon_lower = epsilon / SC(1000 * target_size);
    int iteration = 0;

#ifdef LAP_QUIET
    AuctionOneWay<SC> auction;
#else
    long long hit_count = 0LL;
    long long miss_count = 0LL;
    long long total_hit_count = 0LL;
    long long total_miss_count = 0LL;
    AuctionOneWay<SC> auction(hit_count, miss_count, total_hit_count, total_miss_count);
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
      auction::displayTime(start_time, ss.str().c_str(), std::cout);
#endif
      while (completed < target_size)
      {
#ifndef LAP_QUIET
        std::stringstream ss;
        ss << " eps = " << epsilon;
        auction::displayProgress(start_time, elapsed, completed, target_size, ss.str().c_str(), iteration);
#endif
        iteration++;

        if (epsilon != 0.0f)
        {
#ifdef DEFER_MISS
          completed = auction.template auction<false>(coupling, beta, iterator, find, u_count, unassigned, d_count, deferred, r_count, receiving, B, target_size, epsilon);
#else
          completed = auction.template auction<false>(coupling, beta, iterator, find, u_count, unassigned, r_count, receiving, B, target_size, epsilon);
#endif
        }
        else
        {
#ifdef DEFER_MISS
          completed = auction.template auction<true>(coupling, beta, iterator, find, u_count, unassigned, d_count, deferred, r_count, receiving, B, target_size, epsilon);
#else
          completed = auction.template auction<true>(coupling, beta, iterator, find, u_count, unassigned, r_count, receiving, B, target_size, epsilon);
#endif
        }

#ifndef LAP_QUIET
        if (completed == target_size)
        {
          for (int x = 0; x < dim; x++)
          {
            if (coupling[x] != -1) rowsol[coupling[x]] = x;
          }
          SC ccost = cost<SC>(dim, dim2, costfunc, rowsol);
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
        max_beta = beta[0];
        for (int y = 1; y < target_size; y++) max_beta = std::max(max_beta, beta[y]);
        for (int y = 0; y < target_size; y++) beta[y] -= max_beta;
        find.fixBeta(max_beta);
      }
    }
    for (int x = 0; x < dim2; x++)
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
      solve(dim, dim2, costfunc, iterator, find, beta, rowsol, use_epsilon);
    } else {
      FindLinear<I, SC> find(dim);
      solve(dim, dim2, costfunc, iterator, find, beta, rowsol, use_epsilon);
    }
  }

  // shortcut for square problems
  template <class SC, class CF, class I>
  void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon, bool find_caching)
  {
    solve<SC>(dim, dim, costfunc, iterator, rowsol, use_epsilon, find_caching);
  }

  template <class SC, class CF>
  SC cost(int dim, int dim2, CF &costfunc, int *rowsol)
  {
    SC total = SC(0);
    for (int i = 0; i < dim; i++) total += costfunc.getCost(i, rowsol[i]);
    return total;
  }

  template <class SC, class CF>
  SC cost(int dim, CF &costfunc, int *rowsol)
  {
    return cost<SC, CF>(dim, dim, costfunc, rowsol);
  }
}

