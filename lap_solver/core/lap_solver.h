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

namespace lap
{
#ifndef LAP_QUIET
	class AllocationLogger
	{
		std::deque<void *> allocated;
		std::deque<unsigned long long> size;
		std::deque<char *> alloc_file;
		std::deque<int> alloc_line;
		unsigned long long peak;
		unsigned long long current;
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
		AllocationLogger() { peak = current = (unsigned long long)0; }
		~AllocationLogger() {}
		void destroy()
		{
			lapInfo << "Peak memory usage:" << commify(peak) << " bytes" << std::endl;
			if (allocated.empty()) return;
			lapInfo << "Memory leak list:" << std::endl;
			while (!allocated.empty())
			{
				lapInfo << "  leaked " << commify(size.front()) << " bytes at " << std::hex << allocated.front() << std::dec << ": " << alloc_file.front() << ":" << alloc_line.front() << std::endl;
				size.pop_front();
				allocated.pop_front();
				alloc_file.pop_front();
				alloc_line.pop_front();
			}
		}

		template <class T>
		void free(T a)
		{
			std::lock_guard<std::mutex> guard(lock);
#ifdef LAP_DEBUG
#ifndef LAP_NO_MEM_DEBUG
			lapDebug << "Freeing memory at " << std::hex << (size_t)a << std::dec << std::endl;
#endif
#endif
			for (unsigned long long i = 0; i < allocated.size(); i++)
			{
				if ((void *)a == allocated[i])
				{
					current -= size[i];
					allocated[i] = allocated.back();
					allocated.pop_back();
					size[i] = size.back();
					size.pop_back();
					alloc_line[i] = alloc_line.back();
					alloc_line.pop_back();
					alloc_file[i] = alloc_file.back();
					alloc_file.pop_back();
					return;
				}
			}
		}

		template <class T>
		void alloc(T a, unsigned long long s, const char *file, const int line)
		{
			std::lock_guard<std::mutex> guard(lock);
#ifdef LAP_DEBUG
#ifndef LAP_NO_MEM_DEBUG
			lapDebug << "Allocating " << s * sizeof(T) << " bytes at " << std::hex << (size_t)a << std::dec << " \"" << file << ":" << line << std::endl;
#endif
#endif
			current += s * sizeof(T);
			peak = std::max(peak, current);
			allocated.push_back((void *)a);
			size.push_back(s * sizeof(T));
			alloc_file.push_back((char *)file);
			alloc_line.push_back(line);
		}
	};

	static AllocationLogger allocationLogger;
#endif

	template <typename T>
	void alloc(T * &ptr, unsigned long long width, const char *file, const int line)
	{
		ptr = new T[width]; // this one is allowed
#ifndef LAP_QUIET
		allocationLogger.alloc(ptr, width, file, line);
#endif
	}

	template <typename T>
	void free(T *&ptr)
	{
		if (ptr == (T *)NULL) return;
#ifndef LAP_QUIET
		allocationLogger.free(ptr);
#endif
		delete[] ptr; // this one is allowed
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

	template <class SC, typename COST>
	void getMinMaxBest(SC &min_cost_l, SC &max_cost_l, SC &picked_cost_l, int &j_min, COST &cost, int *taken, int count)
	{
		min_cost_l = max_cost_l = cost(0);
		if (taken[0] == 0)
		{
			j_min = 0;
			picked_cost_l = min_cost_l;
		}
		else
		{
			j_min = std::numeric_limits<int>::max();
			picked_cost_l = std::numeric_limits<SC>::max();
		}
		for (int j = 1; j < count; j++)
		{
			SC cost_l = cost(j);
			min_cost_l = std::min(min_cost_l, cost_l);
			max_cost_l = std::max(max_cost_l, cost_l);
			if ((cost_l < picked_cost_l) && (taken[j] == 0))
			{
				j_min = j;
				picked_cost_l = cost_l;
			}
		}
	}

	template <class SC, typename COST>
	void getMinSecondBest(SC &min_cost_l, SC &second_cost_l, SC &picked_cost_l, int &j_min, COST &cost, int *taken, int count)
	{
		min_cost_l = std::min(cost(0), cost(1));
		second_cost_l = std::max(cost(0), cost(1));
		if ((taken[0] == 0) && (taken[1] == 0))
		{
			picked_cost_l = min_cost_l;
			if (cost(0) == min_cost_l)
			{
				j_min = 0;
			}
			else
			{
				j_min = 1;
			}
		}
		else if (taken[0] == 0)
		{
			j_min = 0;
			picked_cost_l = cost(0);
		}
		else if (taken[1] == 0)
		{
			j_min = 1;
			picked_cost_l = cost(1);
		}
		else
		{
			j_min = std::numeric_limits<int>::max();
			picked_cost_l = std::numeric_limits<SC>::max();
		}
		for (int j = 2; j < count; j++)
		{
			SC cost_l = cost(j);
			if (cost_l < min_cost_l)
			{
				second_cost_l = min_cost_l;
				min_cost_l = cost_l;
			}
			else second_cost_l = std::min(second_cost_l, cost_l);
			if ((cost_l < picked_cost_l) && (taken[j] == 0))
			{
				j_min = j;
				picked_cost_l = cost_l;
			}
		}
	}

	template <class SC, typename COST>
	void updateEstimatedV(SC* v, SC *min_v, COST &cost, bool first, bool second, SC min_cost_l, SC max_cost_l, int count)
	{
		if (first)
		{
			for (int j = 0; j < count; j++)
			{
				SC tmp = cost(j) - min_cost_l;
				min_v[j] = tmp;
			}
		}
		else if (second)
		{
			for (int j = 0; j < count; j++)
			{
				SC tmp = cost(j) - min_cost_l;
				if (tmp < min_v[j])
				{
					v[j] = min_v[j];
					min_v[j] = tmp;
				}
				else v[j] = tmp;
			}
		}
		else
		{
			for (int j = 0; j < count; j++)
			{
				SC tmp = cost(j) - min_cost_l;
				if (tmp < min_v[j])
				{
					v[j] = min_v[j];
					min_v[j] = tmp;
				}
				else v[j] = std::min(v[j], tmp);
			}
		}
	}

	template <class SC>
	void normalizeV(SC *v, int count)
	{
		SC max_v = v[0];
		for (int j = 1; j < count; j++) max_v = std::max(max_v, v[j]);
		for (int j = 0; j < count; j++) v[j] = v[j] - max_v;
	}

	template <class SC, typename COST>
	void getMinimalCost(int &j_min, SC &min_cost, SC &min_cost2, SC &min_cost_real, COST &cost, SC *mod_v, int count)
	{
		j_min = std::numeric_limits<int>::max();
		min_cost = std::numeric_limits<SC>::max();
		min_cost2 = std::numeric_limits<SC>::max();
		min_cost_real = std::numeric_limits<SC>::max();
		for (int j = 0; j < count; j++)
		{
			SC cost_l = cost(j);
			if (mod_v[j] < SC(0))
			{
				if (cost_l < min_cost)
				{
					min_cost = cost_l;
					j_min = j;
				}
			}
			else
			{
				min_cost2 = std::min(min_cost2, cost_l + mod_v[j]);
			}
			min_cost_real = std::min(min_cost_real, cost_l);
		}
	}

	template <class SC, typename COST>
	void updateModV(SC *mod_v, COST &cost, int i, int *picked, SC *update, SC *capacity)
	{
		for (int ii = 0; ii < i; ii++)
		{
			int j = picked[ii];
			SC cost_l = cost(j);
			update[j] = std::max(SC(0), cost_l);
		}
		if (i > 0)
		{
			SC up = update[picked[i - 1]];
			mod_v[picked[i - 1]] += up;
			for (int ii = i - 2; ii >= 0; --ii)
			{
				int j = picked[ii];
				SC up_new = std::max(up - capacity[j], update[j]);
				mod_v[j] += up_new;
				capacity[j] += up_new - up;
				up = up_new;
			}
		}
	}

	template <class SC, class I>
	std::pair<SC, SC> estimateEpsilon(int dim, int dim2, I& iterator, SC *v)
	{
		SC *mod_v;
		int *perm;
		int *picked;
		SC *update;
		SC *capacity;

		lapAlloc(mod_v, dim2, __FILE__, __LINE__);
		lapAlloc(perm, dim, __FILE__, __LINE__);
		lapAlloc(picked, dim2, __FILE__, __LINE__);
		lapAlloc(update, dim2, __FILE__, __LINE__);
		lapAlloc(capacity, dim2, __FILE__, __LINE__);

		SC lower_bound = SC(0);
		SC greedy_bound = SC(0);
		SC upper_bound = SC(0);

		memset(picked, 0, sizeof(int) * dim2);

		for (int i = 0; i < dim; i++)
		{
			SC min_cost_l, max_cost_l, picked_cost_l;
			int j_min;
			const auto *tt = iterator.getRow(i);
			auto cost = [&tt](int j) -> SC { return (SC)tt[j]; };
			getMinMaxBest(min_cost_l, max_cost_l, picked_cost_l, j_min, cost, picked, dim2);
			picked[j_min] = 1;
			updateEstimatedV(v, mod_v, cost, (i == 0), (i == 1), min_cost_l, max_cost_l, dim2);
			lower_bound += min_cost_l;
			upper_bound += (SC)tt[i];
			greedy_bound += picked_cost_l;
		}
		// make sure all j are < 0
		normalizeV(v, dim2);

		greedy_bound = std::min(greedy_bound, upper_bound);

		SC initial_gap = upper_bound - lower_bound;
		SC greedy_gap = greedy_bound - lower_bound;

		SC first_upper_bound = upper_bound;
		SC first_lower_bound = lower_bound;

#ifdef LAP_DEBUG
		lapDebug << "  upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " initial_gap = " << initial_gap << std::endl;
		lapDebug << "  upper_bound = " << greedy_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap << std::endl;
#endif

		SC upper = std::numeric_limits<SC>::max();
		SC lower;

		memset(picked, 0, sizeof(int) * dim2);

		lower_bound = SC(0);
		upper_bound = SC(0);

		// reverse order
		for (int i = dim - 1; i >= 0; --i)
		{
			const auto *tt = iterator.getRow(i);
			SC min_cost_l, second_cost_l, picked_cost_l;
			int j_min;
			auto cost = [&tt, &v](int j) -> SC { return (SC)tt[j] - v[j]; };
			getMinSecondBest(min_cost_l, second_cost_l, picked_cost_l, j_min, cost, picked, dim2);
			picked[j_min] = 1;
			perm[i] = i;
			mod_v[i] = second_cost_l - min_cost_l;
			// need to use the same v values in total
			lower_bound += min_cost_l + v[j_min];
			upper_bound += picked_cost_l + v[j_min];
		}

		upper_bound = greedy_bound = std::min(upper_bound, greedy_bound);

		greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
		lapDebug << "  upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap << std::endl;
#endif
		if (initial_gap < SC(4) * greedy_gap)
		{
			// sort permutation by keys
			std::sort(perm, perm + dim, [&mod_v](int a, int b) { return (mod_v[a] > mod_v[b]) || ((mod_v[a] == mod_v[b]) && (a > b)); });

			lower_bound = SC(0);
			upper_bound = SC(0);
			// greedy search
			std::fill(mod_v, mod_v + dim2, SC(-1));
			for (int i = 0; i < dim; i++)
			{
				// greedy order
				const auto *tt = iterator.getRow(perm[i]);
				int j_min;
				SC min_cost, min_cost2, min_cost_real;
				auto cost = [&tt, &v](int j) -> SC { return (SC)tt[j] - v[j]; };
				getMinimalCost(j_min, min_cost, min_cost2, min_cost_real, cost, mod_v, dim2);
				upper_bound += min_cost + v[j_min];
				// need to use the same v values in total
				lower_bound += min_cost_real + v[j_min];
				SC gap = (i == 0) ? SC(0) : (min_cost - min_cost2);
				if (gap > SC(0))
				{
					auto cost = [&min_cost, &tt, &mod_v, &v](int j) -> SC { return min_cost - ((SC)tt[j] + mod_v[j] - v[j]); };
					updateModV(mod_v, cost, i, picked, update, capacity);
				}
				mod_v[j_min] = SC(0);
				capacity[j_min] = std::max(SC(0), -gap);
				picked[i] = j_min;
			}
			greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
			lapDebug << "  upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap << std::endl;
#endif

			SC max_mod = mod_v[0];
			SC min_mod = mod_v[0];
			for (int j = 1; j < dim2; j++)
			{
				max_mod = std::max(max_mod, mod_v[j]);
				min_mod = std::min(min_mod, mod_v[j]);
			}

			if ((max_mod - min_mod != 0) && (initial_gap < SC(4) * greedy_gap))
			{
				for (int j = 0; j < dim2; j++) v[j] = v[j] - mod_v[j];
				normalizeV(v, dim2);
			}

		}

		if (initial_gap < SC(4) * greedy_gap)
		{
			upper_bound = SC(0);
			lower_bound = SC(0);
			for (int i = dim - 1; i >= 0; --i)
			{
				// reverse greedy order
				const auto *tt = iterator.getRow(perm[i]);
				SC min_cost = (SC)tt[picked[i]];
				SC min_cost_real = std::numeric_limits<SC>::max();
				for (int j = 0; j < dim2; j++)
				{
					SC cost_l = (SC)tt[j] - v[j];
					min_cost_real = std::min(min_cost_real, cost_l);// +v[j]);
				}
				// need to use all picked v for the lower bound as well
				upper_bound += min_cost;
				lower_bound += min_cost_real + v[picked[i]];
			}
			greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
			lapDebug << "  upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap << std::endl;
#endif
		}

		if ((double)greedy_gap <= 1e-6 * (double)initial_gap)
		{
			upper = SC(0);
		}
		else
		{
			double rate_lower = (double)(lower_bound - first_lower_bound);
			double rate_upper = (double)(upper_bound - first_upper_bound);

			double crossing = (double)(first_upper_bound - first_lower_bound) / (rate_lower - rate_upper);

			SC estimate = (SC)((double)first_upper_bound + rate_upper * sqrt(crossing));
#ifdef LAP_DEBUG
			lapDebug << "  crossing = " << crossing << " estimate = " << estimate << std::endl;
#endif

			upper = (estimate - lower_bound) / (SC)(2 * dim2);
		}

		lower = upper / (SC)(dim2 * dim2);

		lapFree(mod_v);
		lapFree(perm);
		lapFree(picked);
		lapFree(update);
		lapFree(capacity);

		return std::pair<SC, SC>((SC)upper, (SC)lower);
	}

#if defined(__GNUC__)
#define __forceinline \
        __inline__ __attribute__((always_inline))
#endif

	__forceinline void dijkstraCheck(int &endofpath, bool &unassignedfound, int jmin, int *colsol, char *colactive, int *colcomplete, int &completecount)
	{
		colactive[jmin] = 0;
		colcomplete[completecount++] = jmin;
		if (colsol[jmin] < 0)
		{
			endofpath = jmin;
			unassignedfound = true;
		}
	}

	template <class SC>
	__forceinline void updateColumnPrices(int *colcomplete, int completecount, SC min, SC *v, SC *d)
	{
		for (int i = 0; i < completecount; i++)
		{
			int j1 = colcomplete[i];
			SC dlt = min - d[j1];
			v[j1] -= dlt;
		}
	}

	template <class SC>
	__forceinline void updateColumnPrices(int *colcomplete, int completecount, SC min, SC *v, SC *d, SC eps, SC &total, SC &total_eps)
	{
		for (int i = 0; i < completecount; i++)
		{
			int j1 = colcomplete[i];
			SC dlt = min - d[j1];
			total -= dlt;
			dlt += eps;
			total_eps -= dlt;
			v[j1] -= dlt;
		}
	}

	template <class SC>
	__forceinline void updateColumnPricesClamp(int *colcomplete, int completecount, SC min, SC *v, SC *d, SC eps, SC &total, SC &total_eps)
	{
		for (int i = 0; i < completecount; i++)
		{
			int j1 = colcomplete[i];
			SC dlt = min - d[j1];
			total -= dlt;
			dlt = std::max(dlt, eps);
			total_eps -= dlt;
			v[j1] -= dlt;
		}
	}

	__forceinline void resetRowColumnAssignment(int &endofpath, int f, int *pred, int *rowsol, int *colsol)
	{
		int i;
		do
		{
			i = pred[endofpath];
			colsol[endofpath] = i;
			int j1 = endofpath;
			endofpath = rowsol[i];
			rowsol[i] = j1;
		} while (i != f);
	}

	template <class SC>
	void getNextEpsilon(SC &epsilon, SC &epsilon_lower, SC total_d, SC total_eps, bool first, int dim2)
	{
		total_eps = total_d - total_eps;
		total_d = -total_d;
		if (epsilon > SC(0))
		{
			if (!first)
			{
#ifdef LAP_DEBUG
				lapDebug << "  v_d = " << total_d / SC(dim2) << " v_eps = " << total_eps / SC(dim2) << " eps = " << epsilon;
#endif
				double v_d = (double)total_d / (double)dim2;
				double v_eps = (double)total_eps / (double)dim2;
				//if (((epsilon * 4 * dim2 > total_eps) && (total_d > total_eps)) || (total_d > SC(16) * total_eps))
				if ((v_d > 0.0) && (v_eps * v_eps / v_d < (double)epsilon))
				{
					epsilon = SC(0);
				}
				else
				{
					epsilon = std::min(epsilon / SC(4), total_eps / SC(8 * dim2));
				}
#ifdef LAP_DEBUG
				lapDebug << " -> " << epsilon;
#endif
				if (epsilon < epsilon_lower)
				{
					epsilon = SC(0);
				}
#ifdef LAP_DEBUG
				lapDebug << " -> " << epsilon << std::endl;
#endif
			}
		}
	}

	template <class SC, class CF, class I>
	void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)

		// input:
		// dim        - problem size
		// costfunc - cost matrix
		// findcost   - searching cost matrix

		// output:
		// rowsol     - column assigned to row in solution
		// colsol     - row assigned to column in solution
		// u          - dual variables, row reduction numbers
		// v          - dual variables, column reduction numbers

	{
#ifndef LAP_QUIET
		auto start_time = std::chrono::high_resolution_clock::now();

		long long total_hit = 0LL;
		long long total_miss = 0LL;

		long long total_rows = 0LL;
		long long total_virtual = 0LL;

		long long last_rows = 0LL;
		long long last_virtual = 0LL;

		int elapsed = -1;
#else
#ifdef LAP_DISPLAY_EVALUATED
		long long total_hit = 0LL;
		long long total_miss = 0LL;

		long long total_rows = 0LL;
		long long total_virtual = 0LL;
#endif
#endif

		int  *pred;
		int  endofpath;
		char *colactive;
		int *colcomplete;
		int completecount;
		SC *d;
		int *colsol;
		SC epsilon_upper;
		SC epsilon_lower;
		SC *v;

#ifdef LAP_DEBUG
		std::vector<SC *> v_list;
		std::vector<SC> eps_list;
#endif

		lapAlloc(colactive, dim2, __FILE__, __LINE__);
		lapAlloc(colcomplete, dim2, __FILE__, __LINE__);
		lapAlloc(d, dim2, __FILE__, __LINE__);
		lapAlloc(pred, dim2, __FILE__, __LINE__);
		lapAlloc(colsol, dim2, __FILE__, __LINE__);
		lapAlloc(v, dim2, __FILE__, __LINE__);

#ifdef LAP_ROWS_SCANNED
		unsigned long long *scancount;
		unsigned long long *pathlength;
		lapAlloc(scancount, dim2, __FILE__, __LINE__);
		lapAlloc(pathlength, dim2, __FILE__, __LINE__);
		memset(scancount, 0, dim2 * sizeof(unsigned long long));
		memset(pathlength, 0, dim2 * sizeof(unsigned long long));
#endif

		SC epsilon;

		if (use_epsilon)
		{
			std::pair<SC, SC> eps = estimateEpsilon(dim, dim2, iterator, v);
			epsilon_upper = eps.first;
			epsilon_lower = eps.second;
		}
		else
		{
			memset(v, 0, dim2 * sizeof(SC));
			epsilon_upper = SC(0);
			epsilon_lower = SC(0);
		}
		epsilon = epsilon_upper;

		bool first = true;
		bool clamp = true;

		SC total_d = SC(0);
		SC total_eps = SC(0);
		while (epsilon >= SC(0))
		{
#ifdef LAP_DEBUG
			if (first)
			{
				SC *vv;
				lapAlloc(vv, dim2, __FILE__, __LINE__);
				v_list.push_back(vv);
				eps_list.push_back(epsilon);
				memcpy(v_list.back(), v, sizeof(SC) * dim2);
			}
#endif
			getNextEpsilon(epsilon, epsilon_lower, total_d, total_eps, first, dim2);

			total_d = SC(0);
			total_eps = SC(0);
#ifndef LAP_QUIET
			{
				std::stringstream ss;
				ss << "eps = " << epsilon;
				const std::string tmp = ss.str();
				displayTime(start_time, tmp.c_str(), lapInfo);
			}
#endif
			// this is to ensure termination of the while statement
			if (epsilon == SC(0)) epsilon = SC(-1.0);
			memset(rowsol, -1, dim2 * sizeof(int));
			memset(colsol, -1, dim2 * sizeof(int));
			int jmin, jmin_n;
			SC min, min_n;
			bool unassignedfound;

#ifndef LAP_QUIET
			int old_complete = 0;
#endif

			// AUGMENT SOLUTION for each free row.
#ifndef LAP_QUIET
			displayProgress(start_time, elapsed, 0, dim2, " rows");
#endif
			int dim_limit = ((epsilon > SC(0)) && (first)) ? dim : dim2;
			for (int f = 0; f < dim_limit; f++)
			{
#ifndef LAP_QUIET
				if (f < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
				if (f < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
				scancount[f]++;
#endif

				unassignedfound = false;
				completecount = 0;

				// Dijkstra search
				min = std::numeric_limits<SC>::max();
				jmin = dim2;
				if (f < dim)
				{
					auto tt = iterator.getRow(f);
					for (int j = 0; j < dim2; j++)
					{
						colactive[j] = 1;
						pred[j] = f;
						SC h = d[j] = tt[j] - v[j];
						if (h < min)
						{
							// better
							jmin = j;
							min = h;
						}
						else if (h == min)
						{
							// same, do only update if old was used and new is free
							if ((colsol[jmin] >= 0) && (colsol[j] < 0)) jmin = j;
						}
					}
				}
				else
				{
					for (int j = 0; j < dim2; j++)
					{
						colactive[j] = 1;
						pred[j] = f;
						SC h = d[j] = -v[j];
						if (colsol[j] < dim)
						{
							if (h < min)
							{
								// better
								jmin = j;
								min = h;
							}
							else if (h == min)
							{
								// same, do only update if old was used and new is free
								if ((colsol[jmin] >= 0) && (colsol[j] < 0)) jmin = j;
							}
						}
					}
				}

				dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive, colcomplete, completecount);
				// marked skipped columns that were cheaper
				if (f >= dim)
				{
					for (int j = 0; j < dim2; j++)
					{
						// ignore any columns assigned to virtual rows
						if ((colsol[j] >= dim) && (d[j] <= min))
						{
							colcomplete[completecount++] = j;
							colactive[j] = 0;
						}
					}
				}

				while (!unassignedfound)
				{
					// update 'distances' between freerow and all unscanned columns, via next scanned column.
					int i = colsol[jmin];
#ifndef LAP_QUIET
					if (i < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
					if (i < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
					scancount[i]++;
#endif

					jmin_n = dim2;
					min_n = std::numeric_limits<SC>::max();
					if (i < dim)
					{
						auto tt = iterator.getRow(i);
						//SC h2 = tt[jmin] - v[jmin] - min;
						SC tt_jmin = (SC)tt[jmin];
						SC v_jmin = v[jmin];
						for (int j = 0; j < dim2; j++)
						{
							if (colactive[j] != 0)
							{
								//SC v2 = tt[j] - v[j] - h2;
								SC v2 = (tt[j] - tt_jmin) - (v[j] - v_jmin) + min;
								SC h = d[j];
								if (v2 < h)
								{
									pred[j] = i;
									d[j] = v2;
									h = v2;
								}
								if (h < min_n)
								{
									// better
									jmin_n = j;
									min_n = h;
								}
								else if (h == min_n)
								{
									// same, do only update if old was used and new is free
									if ((colsol[jmin_n] >= 0) && (colsol[j] < 0)) jmin_n = j;
								}
							}
						}
					}
					else
					{
						//SC h2 = -v[jmin] - min;
						SC v_jmin = v[jmin];
						for (int j = 0; j < dim2; j++)
						{
							if (colactive[j] != 0)
							{
								//SC v2 = -v[j] - h2;
								SC v2 = -(v[j] - v_jmin) + min;
								SC h = d[j];
								if (v2 < h)
								{
									pred[j] = i;
									d[j] = v2;
									h = v2;
								}
								if (colsol[j] < dim)
								{
									if (h < min_n)
									{
										// better
										jmin_n = j;
										min_n = h;
									}
									else if (h == min_n)
									{
										// same, do only update if old was used and new is free
										if ((colsol[jmin_n] >= 0) && (colsol[j] < 0)) jmin_n = j;
									}
								}
							}
						}
					}

					min = std::max(min, min_n);
					jmin = jmin_n;
					dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive, colcomplete, completecount);
					// marked skipped columns that were cheaper
					if (i >= dim)
					{
						for (int j = 0; j < dim2; j++)
						{
							// ignore any columns assigned to virtual rows
							if ((colactive[j] == 1) && (colsol[j] >= dim) && (d[j] <= min_n))
							{
								colcomplete[completecount++] = j;
								colactive[j] = 0;
							}
						}
					}
				}

				// update column prices. can increase or decrease
				if (epsilon > SC(0))
				{
					if (clamp) updateColumnPricesClamp(colcomplete, completecount, min, v, d, epsilon, total_d, total_eps);
					else updateColumnPrices(colcomplete, completecount, min, v, d, epsilon, total_d, total_eps);
				}
				else
				{
					updateColumnPrices(colcomplete, completecount, min, v, d);
				}
#ifdef LAP_ROWS_SCANNED
				{
					int i;
					int eop = endofpath;
					do
					{
						i = pred[eop];
						eop = rowsol[i];
						if (i != f) pathlength[f]++;
					} while (i != f);
				}
#endif

				// reset row and column assignments along the alternating path.
				resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
#ifndef LAP_QUIET
				int level;
				if ((level = displayProgress(start_time, elapsed, f + 1, dim2, " rows")) != 0)
				{
					long long hit, miss;
					iterator.getHitMiss(hit, miss);
					total_hit += hit;
					total_miss += miss;
					if ((hit != 0) || (miss != 0))
					{
						if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
						else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
					}
					old_complete = f + 1;
				}
#endif
			}

			if (dim_limit < dim2)
			{
				total_eps -= SC(dim2 - dim_limit) * epsilon;
				// fix v in unassigned columns
				for (int j = 0; j < dim2; j++)
				{
					if (colsol[j] < 0) v[j] -= epsilon;
				}
			}

#ifdef LAP_MINIMIZE_V
			if (epsilon > SC(0))
			{
				normalizeV(v, dim2);
			}
#endif

#ifdef LAP_DEBUG
			if (epsilon > SC(0))
			{
				SC *vv;
				lapAlloc(vv, dim2, __FILE__, __LINE__);
				v_list.push_back(vv);
				eps_list.push_back(epsilon);
				memcpy(v_list.back(), v, sizeof(SC) * dim2);
			}
			else
			{
				int count = (int)v_list.size();
				if (count > 0)
				{
					for (int l = 0; l < count; l++)
					{
						SC dlt(0), dlt2(0);
						for (int i = 0; i < dim2; i++)
						{
							SC diff = v_list[l][i] - v[i];
							dlt += diff;
							dlt2 += diff * diff;
						}
						dlt /= SC(dim2);
						dlt2 /= SC(dim2);
						lapDebug << "iteration = " << l << " eps/mse = " << eps_list[l] << " " << dlt2 - dlt * dlt << " eps/rmse = " << eps_list[l] << " " << sqrt(dlt2 - dlt * dlt) << std::endl;
						lapFree(v_list[l]);
					}
				}
			}
#endif
			first = false;
#ifndef LAP_QUIET
			lapInfo << "  rows evaluated: " << total_rows;
			if (last_rows > 0LL) lapInfo << " (+" << total_rows - last_rows << ")";
			last_rows = total_rows;
			if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
			if (last_virtual > 0LL) lapInfo << " (+" << total_virtual - last_virtual << ")";
			last_virtual = total_virtual;
			lapInfo << std::endl;
			if ((total_hit != 0) || (total_miss != 0)) lapInfo << "  hit: " << total_hit << " miss: " << total_miss << std::endl;
#endif
		}

#ifdef LAP_QUIET
#ifdef LAP_DISPLAY_EVALUATED
		iterator.getHitMiss(total_hit, total_miss);
		lapInfo << "  rows evaluated: " << total_rows;
		if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
		lapInfo << std::endl;
		if ((total_hit != 0) || (total_miss != 0)) lapInfo << "  hit: " << total_hit << " miss: " << total_miss << std::endl;
#endif
#endif

#ifdef LAP_ROWS_SCANNED
		for (int f = 0; f < dim2; f++)
		{
			lapInfo << "row: " << f << " scanned: " << scancount[f] << " length: " << pathlength[f] << std::endl;
		}

		lapFree(scancount);
		lapFree(pathlength);
#endif

#ifdef LAP_VERIFY_RESULT
		SC slack = SC(0);
		bool correct = true;
		for (int f = 0; f < dim2; f++)
		{
			auto tt = iterator.getRow(f);
			int jmin = rowsol[f];
			SC ref_min = tt[jmin] - v[jmin];
			SC min = ref_min;
			for (int j = 0; j < dim2; j++)
			{
				SC h = tt[j] - v[j];
				if (h < min)
				{
					// better
					jmin = j;
					min = h;
				}
			}
			if (jmin != rowsol[f])
			{
				slack += ref_min - min;
				correct = false;
			}
		}
		if (correct)
		{
			lapInfo << "Solution accurate." << std::endl;
		}
		else
		{
			lapInfo << "Solution might be inaccurate (slack = " << slack << ")." << std::endl;
		}
#endif

		// free reserved memory.
		lapFree(pred);
		lapFree(colactive);
		lapFree(colcomplete);
		lapFree(d);
		lapFree(v);
		lapFree(colsol);
	}

	// shortcut for square problems
	template <class SC, class CF, class I>
	void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)
	{
		solve<SC>(dim, dim, costfunc, iterator, rowsol, use_epsilon);
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
