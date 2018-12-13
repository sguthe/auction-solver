#pragma once

#include <tuple>
#include <vector>
#include <thread>
#include <omp.h>
#include "auction_heap.h"

template <class COST, class AC>
class FindLinear
{
private:
	int target_size;
public:
	FindLinear(int target_size) : target_size(target_size) {}

	template <bool PAR>
	__forceinline__	bool findBid(COST &c, int x, std::pair<int, int> &y, std::pair<AC, AC> &cost, std::vector<AC> &beta)
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
	__forceinline__ void fillCache(COST &c, int x, std::pair<int, int> &y, std::pair<AC, AC> &cost, std::vector<AC> &beta) {}
	// nothing here
	__forceinline__ void fixBeta(AC dlt) {}
};

template <class COST, class AC>
class FindLinearMultiple : public FindLinear<COST, AC>
{
public:
	FindLinearMultiple(int target_size) : FindLinear<COST, AC>::FindLinear(target_size) {}

	template <bool PAR>
	__forceinline__ bool findBid(COST &c, int x, std::pair<int, int> &y, std::pair<AC, AC> &cost, std::vector<AC> &beta) { return FindLinear<COST, AC>::template findBid<PAR>(c, x, y, cost, beta); }

	template <bool PAR>
	__forceinline__	bool findBid(COST &c, int x, std::vector<int> &y, std::pair<AC, AC> &cost, std::vector<AC> &beta, std::vector<int> &coupling)
	{
		//y.resize(count);
		//cost.resize(count);
		y.clear();
		cost.first = cost.second = MAX_COST;
		//int N = 0;
		// y.1 = argmin(...) and y.2 = min(...) with y.2 in Y\y.1
		//int yb = 0;
		c.template iterateMulti<PAR>([&](int yy, AC ccost)
		{
			if (PAR)
			{
				//if (coupling[yy] != x)
				{
					if (ccost <= cost.second)
					{
#pragma omp critical
						{
							if ((ccost - cost.first) < -1e-9f)
							{
								y.clear();
								y.push_back(yy);
								cost.second = cost.first;
								cost.first = ccost;
							}
							else if ((ccost - cost.first) < 1e-9f)
							{
								y.push_back(yy);
							}
							else if ((ccost - cost.second) < -1e-9f)
							{
								cost.second = ccost;
							}
						}
					}
				}
			}
			else
			{
				//if (coupling[yy] != x)
				{
					if ((ccost - cost.first) < -1e-9f)
					{
						y.clear();
						y.push_back(yy);
						cost.second = cost.first;
						cost.first = ccost;
					}
					else if ((ccost - cost.first) < 1e-9f)
					{
						y.push_back(yy);
					}
					else if ((ccost - cost.second) < -1e-9f)
					{
						cost.second = ccost;
					}
				}
			}
		}, x, cost.second, beta);
		return true;
	}
	// nothing here
	__forceinline__ void fixBeta(AC dlt) {}
	// nothing here
	__forceinline__ void restart() {}
};
