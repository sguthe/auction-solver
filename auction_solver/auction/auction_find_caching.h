#pragma once

#include <tuple>
#include <vector>

#include "auction_helper.h"
#include "auction_heap.h"

extern int processor_count;

template <class COST, int CACHE, class AC>
class FindCaching
{
protected:
	int target_size;
	std::vector<std::vector<int>> m_idx;
	std::vector<std::vector<AC>> m_heap;

//	std::vector<AC> temp_cost;
	std::vector<std::vector<int>> temp_idx;
	std::vector<std::vector<AC>> temp_heap;
public:
	FindCaching(int target_size, COST &c, std::vector<AC> &beta, bool fill = true) : target_size(target_size)
	{
#ifdef DISPLAY_THREAD_FILL
		thread_fill_count.resize(omp_get_max_threads());
#endif
		//temp_cost.resize(target_size);
		temp_idx.resize(processor_count);
		temp_heap.resize(processor_count);
		for (int x = 0; x < processor_count; x++)
		{
			temp_idx[x].resize(CACHE + 1);
			temp_heap[x].resize(CACHE + 1);
		}

		m_idx.resize(target_size);
		m_heap.resize(target_size);
		for (int x = 0; x < target_size; x++)
		{
			m_idx[x].resize(CACHE + 1);
			m_heap[x].resize(CACHE + 1);
		}
		if (fill)
		{
			bool hacky_hack = false;
			if (beta.size() == 0)
			{
				beta.resize(target_size);
				for (int x = 0; x < target_size; x++) beta[x] = AC(0.0);
				hacky_hack = true;
			}
#pragma omp parallel
			{
				int t = omp_get_thread_num();
#pragma omp for schedule(dynamic)
				for (int x = 0; x < target_size; x++)
				{
					std::pair<int, int> y;
					std::pair<AC, AC> cost;
					fillCache<false, true>(c, x, y, cost, beta);
				}
			}
			c.printCases(true);
			if (hacky_hack) beta.resize(0);
		}
	}

	void recreate(COST &c, float *linear_target)
	{
		omp_set_num_threads(processor_count);
#pragma omp parallel for
		for (int x = 0; x < target_size; x++)
		{
			float mod = c.update_target(linear_target, x);
			std::vector<int> &idx = m_idx[x];
			std::vector<AC> &heap = m_heap[x];
			for (int yy = 0; yy < CACHE; yy++)
			{
				int y = idx[yy];
				heap[yy] = c.getCost(x, y);
			}
			mod = sqrtf(heap[CACHE]) - sqrtf(mod);
			heap[CACHE] = mod * mod;
			//heap[CACHE] = -1.0e36f;
		}
	}

	~FindCaching() {}

	// this doesn't have to be the same class
	template <bool PAR, bool FILL_ONLY>
	void fillCache(COST &c, int x, std::pair<int, int> &y, std::pair<AC, AC> &cost, std::vector<AC> &beta)
	{
		std::vector<int> &idx = m_idx[x];
		std::vector<AC> &heap = m_heap[x];

#ifdef DISPLAY_THREAD_FILL
		thread_fill_count[omp_get_thread_num()]++;
#endif

		AC limit = MAX_COST;
		int N = 0;
		if (PAR)
		{
			std::vector<int> NN(processor_count);
			c.template iterate<PAR>([&](int yy, AC ccost)
			{
				if (ccost <= limit)
				{
					int p = omp_get_thread_num();
					if (NN[p] < CACHE + 1)
					{
						heap_insert(temp_heap[p], temp_idx[p], NN[p], ccost, yy);
					}
					else
					{
						heap_replace(temp_heap[p], temp_idx[p], NN[p], ccost, yy);
						if (temp_heap[p][0] < limit)
						{
#pragma omp critical
							limit = std::min(limit, temp_heap[p][0]);
						}
					}
				}
			}, x, limit, beta);

			for (int p = 0; p < processor_count; p++)
			{
				for (int yy = 0; yy < NN[p]; yy++)
				{
					if (temp_heap[p][yy] <= limit)
					{
						if (N < CACHE + 1)
						{
							heap_insert(heap, idx, N, temp_heap[p][yy], temp_idx[p][yy]);
						}
						else
						{
							heap_replace(heap, idx, N, temp_heap[p][yy], temp_idx[p][yy]);
							limit = heap[0];
						}
					}
				}
			}
			// for a heap the max is at 0 but we need it at the end.
			std::swap(heap[0], heap[CACHE]);
			std::swap(idx[0], idx[CACHE]);
		}
		else
		{
			c.template iterate<PAR>([&](int yy, AC ccost)
			{
				if (N < CACHE + 1)
				{
					heap_insert(heap, idx, N, ccost, yy);
				}
				else
				{
					heap_replace(heap, idx, N, ccost, yy);
					limit = heap[0];
				}
			}, x, limit, beta);
			// for a heap the max is at 0 but we need it at the end.
			std::swap(heap[0], heap[CACHE]);
			std::swap(idx[0], idx[CACHE]);
		}


		if (FILL_ONLY)
		{
			if (PAR)
			{
#pragma omp parallel for
				for (int yi = 0; yi < CACHE; yi++)
				{
					heap[yi] = c.getCost(x, idx[yi]);
				}
			}
			else
			{
				for (int yi = 0; yi < CACHE; yi++)
				{
					heap[yi] = c.getCost(x, idx[yi]);
				}
			}
		}
		else
		{
			// scan with beta
			y.first = -1;
			y.second = -1;
			cost.first = cost.second = MAX_COST;
			for (int yi = 0; yi < CACHE; yi++)
			{
				int yy = idx[yi];
				heap[yi] = c.getCost(x, yy);
				AC ccost = heap[yi] - beta[yy];
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

	template <bool PAR>
	bool findBid(COST &c, int x, std::pair<int, int> &y, std::pair<AC, AC> &cost, std::vector<AC> &beta)
	{
		y.first = y.second = -1;
		cost.first = cost.second = MAX_COST;
		// cached
		std::vector<int> &idx = m_idx[x];
		std::vector<AC> &heap = m_heap[x];

		// TODO: doing this in parallel requires reduction by hand...
		for (int yi = 0; yi < CACHE; yi++)
		{
			int yy = idx[yi];
			AC ccost = heap[yi] - beta[yy];
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
#ifdef DEFER_MISS
		if (cost.second <= heap[CACHE]) return true;
		else return false;
#else
		if (PAR)
		{
			if (cost.second <= heap[CACHE])
			{
				hit_count++;
				return true;
			}
			miss_count++;
		}
		else
		{
			if (cost.second <= heap[CACHE])
			{
#pragma omp atomic
				hit_count++;
				return true;
			}
#pragma omp atomic
			miss_count++;
		}

		// cache didn't work so rebuild it
		fillCache<PAR, false>(c, x, y, cost, beta);

		return true;
#endif
	}

	void fixBeta(AC dlt)
	{
#pragma omp parallel for
		for (int x = 0; x < target_size; x++)
		{
			if (!m_heap[x].empty()) m_heap[x][CACHE] += dlt;
		}
	}

	void unblock() { }
	void block() { }
};

template <class COST, int CACHE, class AC>
class FindCachingReverse
{
protected:
	int target_size;
	std::vector<std::vector<int>> m_idx;
	std::vector<std::vector<AC>> m_heap;

	//	std::vector<AC> temp_cost;
	std::vector<std::vector<int>> temp_idx;
	std::vector<std::vector<AC>> temp_heap;
public:
	FindCachingReverse(int target_size, COST &c, std::vector<AC> &beta, bool fill = true) : target_size(target_size)
	{
#ifdef DISPLAY_THREAD_FILL
		thread_fill_count.resize(omp_get_max_threads());
#endif
		//temp_cost.resize(target_size);
		temp_idx.resize(processor_count);
		temp_heap.resize(processor_count);
		for (int x = 0; x < processor_count; x++)
		{
			temp_idx[x].resize(CACHE + 1);
			temp_heap[x].resize(CACHE + 1);
		}

		m_idx.resize(target_size);
		m_heap.resize(target_size);
		for (int x = 0; x < target_size; x++)
		{
			m_idx[x].resize(CACHE + 1);
			m_heap[x].resize(CACHE + 1);
		}
		if (fill)
		{
			bool hacky_hack = false;
			if (beta.size() == 0)
			{
				beta.resize(target_size);
				for (int x = 0; x < target_size; x++) beta[x] = AC(0.0);
				hacky_hack = true;
			}
#pragma omp parallel
			{
#pragma omp for schedule(dynamic)
				for (int x = 0; x < target_size; x++)
				{
					std::pair<int, int> y;
					std::pair<AC, AC> cost;
					fillCache<false, true>(c, x, y, cost, beta);
				}
			}
			c.printCases(true);
			if (hacky_hack) beta.resize(0);
		}
	}

	void recreate(COST &c, float *linear_target)
	{
		omp_set_num_threads(processor_count);
#pragma omp parallel for
		for (int x = 0; x < target_size; x++)
		{
			float mod = c.update_target(linear_target, x);
			std::vector<int> &idx = m_idx[x];
			std::vector<AC> &heap = m_heap[x];
			for (int yy = 0; yy < CACHE; yy++)
			{
				int y = idx[yy];
				heap[yy] = c.getCost(y, x);
			}
			mod = sqrtf(heap[CACHE]) - sqrtf(mod);
			heap[CACHE] = mod * mod;
			//heap[CACHE] = -1.0e36f;
		}
	}

	~FindCachingReverse() {}

	// this doesn't have to be the same class
	template <bool PAR, bool FILL_ONLY>
	void fillCache(COST &c, int x, std::pair<int, int> &y, std::pair<AC, AC> &cost, std::vector<AC> &beta)
	{
		std::vector<int> &idx = m_idx[x];
		std::vector<AC> &heap = m_heap[x];

#ifdef DISPLAY_THREAD_FILL
		thread_fill_count[omp_get_thread_num()]++;
#endif

		AC limit = MAX_COST;
		int N = 0;
		if (PAR)
		{
			std::vector<int> NN(processor_count);
			c.template iterateReverse<PAR>([&](int yy, AC ccost)
			{
				if (ccost <= limit)
				{
					int p = omp_get_thread_num();
					if (NN[p] < CACHE + 1)
					{
						heap_insert(temp_heap[p], temp_idx[p], NN[p], ccost, yy);
					}
					else
					{
						heap_replace(temp_heap[p], temp_idx[p], NN[p], ccost, yy);
						if (temp_heap[p][0] < limit)
						{
#pragma omp critical
							limit = std::min(limit, temp_heap[p][0]);
						}
					}
				}
			}, x, limit, beta);

			for (int p = 0; p < processor_count; p++)
			{
				for (int yy = 0; yy < NN[p]; yy++)
				{
					if (temp_heap[p][yy] <= limit)
					{
						if (N < CACHE + 1)
						{
							heap_insert(heap, idx, N, temp_heap[p][yy], temp_idx[p][yy]);
						}
						else
						{
							heap_replace(heap, idx, N, temp_heap[p][yy], temp_idx[p][yy]);
							limit = heap[0];
						}
					}
				}
			}
			// for a heap the max is at 0 but we need it at the end.
			std::swap(heap[0], heap[CACHE]);
			std::swap(idx[0], idx[CACHE]);
		}
		else
		{
			c.template iterateReverse<PAR>([&](int yy, AC ccost)
			{
				if (N < CACHE + 1)
				{
					heap_insert(heap, idx, N, ccost, yy);
				}
				else
				{
					heap_replace(heap, idx, N, ccost, yy);
					limit = heap[0];
				}
			}, x, limit, beta);
			// for a heap the max is at 0 but we need it at the end.
			std::swap(heap[0], heap[CACHE]);
			std::swap(idx[0], idx[CACHE]);
		}


		if (FILL_ONLY)
		{
			if (PAR)
			{
#pragma omp parallel for
				for (int yi = 0; yi < CACHE; yi++)
				{
					heap[yi] = c.getCost(idx[yi], x);
				}
			}
			else
			{
				for (int yi = 0; yi < CACHE; yi++)
				{
					heap[yi] = c.getCost(idx[yi], x);
				}
			}
		}
		else
		{
			// scan with beta
			y.first = y.second = -1;
			cost.first = cost.second = MAX_COST;
			for (int yi = 0; yi < CACHE; yi++)
			{
				int yy = idx[yi];
				heap[yi] = c.getCost(yy, x);
				AC ccost = heap[yi] - beta[yy];
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

	template <bool PAR>
	bool findBid(COST &c, int x, std::pair<int, int> &y, std::pair<AC, AC> &cost, std::vector<AC> &beta)
	{
		y.first = y.second = -1;
		cost.first = cost.second = MAX_COST;
		// cached
		std::vector<int> &idx = m_idx[x];
		std::vector<AC> &heap = m_heap[x];

		// TODO: doing this in parallel requires reduction by hand...
		for (int yi = 0; yi < CACHE; yi++)
		{
			int yy = idx[yi];
			AC ccost = heap[yi] - beta[yy];
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
#ifdef DEFER_MISS
		if (cost.second <= heap[CACHE]) return true;
		else return false;
#else
		if (PAR)
		{
			if (cost.second <= heap[CACHE])
			{
				hit_count++;
				return true;
			}
			miss_count++;
		}
		else
		{
			if (cost.second <= heap[CACHE])
			{
#pragma omp atomic
				hit_count++;
				return true;
			}
#pragma omp atomic
			miss_count++;
		}

		// cache didn't work so rebuild it
		fillCache<PAR, false>(c, x, y, cost, beta);

		return true;
#endif
	}

	void fixBeta(AC dlt)
	{
#pragma omp parallel for
		for (int x = 0; x < target_size; x++)
		{
			if (!m_heap[x].empty()) m_heap[x][CACHE] += dlt;
		}
	}

	void unblock() { }
	void block() { }
};

