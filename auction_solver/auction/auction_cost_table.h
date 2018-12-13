#pragma once

#include <vector>
#include "auction_helper.h"
#include "auction_cost_multiple.h"

template <class AC>
class TableCost
{
private:
	std::vector<std::vector<AC>> c;
	int target_size;
public:
	TableCost(float *linear_target, int target_size, float *linear_source, int source_size, int total_channel)
		: target_size(target_size)
	{
		c.resize(target_size);
		for (int x = 0; x < target_size; x++) c[x].resize(target_size);
#pragma omp parallel for
		for (int x = 0; x < target_size; x++)
		{
			for (int y = 0; y < target_size; y++)
			{
				int y0 = src_idx(y, source_size, target_size);
				AC cost = AC(0.0);
				for (int cc = 0; cc < total_channel; cc++)
				{
					AC d = linear_source[cc * source_size + y0] - linear_target[cc * target_size + x];
					cost = cost + d * d;
				}
				cost /= (AC)total_channel;
				c[x][y] = cost;
			}
		}
	}
	__forceinline__ const AC getCost(int x, int y) const { return c[x][y]; }

	template <bool PAR, class C>
	__forceinline__ void iterate(const C &c, int x, AC &limit, std::vector<AC> &beta)
	{
		if (PAR)
		{
#pragma omp parallel for
			for (int yy = 0; yy < target_size; yy++)
			{
				if (-beta[yy] <= limit)
				{
					AC ccost = getCost(x, yy) - beta[yy];
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
					AC ccost = getCost(x, yy) - beta[yy];
					c(yy, ccost);
				}
			}
		}
	}
	template <bool PAR, class C>
	__forceinline__ void iterate(const C &c, int x, std::vector<AC> &beta)
	{
		if (PAR)
		{
#pragma omp parallel for
			for (int yy = 0; yy < target_size; yy++)
			{
				AC ccost = getCost(x, yy) - beta[yy];
				c(yy, ccost);
			}
		}
		else
		{
			for (int yy = 0; yy < target_size; yy++)
			{
				AC ccost = getCost(x, yy) - beta[yy];
				c(yy, ccost);
			}
		}
	}
	template <bool PAR, class C>
	__forceinline__ void iterateReverse(const C &c, int x, AC &limit, std::vector<AC> &beta)
	{
		if (PAR)
		{
#pragma omp parallel for
			for (int yy = 0; yy < target_size; yy++)
			{
				if (-beta[yy] <= limit)
				{
					AC ccost = getCost(yy, x) - beta[yy];
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
					AC ccost = getCost(yy, x) - beta[yy];
					c(yy, ccost);
				}
			}
		}
	}
	template <bool PAR, class C>
	__forceinline__ void iterateReverse(const C &c, int x, std::vector<AC> &beta)
	{
		if (PAR)
		{
#pragma omp parallel for
			for (int yy = 0; yy < target_size; yy++)
			{
				AC ccost = getCost(yy, x) - beta[yy];
				c(yy, ccost);
			}
		}
		else
		{
			for (int yy = 0; yy < target_size; yy++)
			{
				AC ccost = getCost(yy, x) - beta[yy];
				c(yy, ccost);
			}
		}
	}
	__forceinline__ void updateBeta(std::vector<AC> &beta, int r_count) {}
	__forceinline__ void updateAlpha(std::vector<AC> &alpha, int r_count) {}
	__forceinline__ void printCases(bool reset) {}
	__forceinline__ bool cudaEnabled(int t) { return false; }
	__forceinline__ void cudaFill(std::vector<int> &idx, std::vector<AC> &heap, int x) {}
	__forceinline__ void cudaFillRow(int t, AC * data, int x, int start, int count) {}
	__forceinline__ void cudaFillReverse(std::vector<int> &idx, std::vector<AC> &heap, int x) {}
	__forceinline__ void cudaFillColumn(int t, AC * data, int x, int start, int count) {}
	__forceinline__ bool requiresCaching() { return false; }
};

template <class AC>
class TableCostMultiple
{
private:
	std::vector<std::vector<AC>> c;
	CostMultiple<AC> &target;
	CostMultiple<AC> &source;

public:
	TableCostMultiple(CostMultiple<AC> &target, CostMultiple<AC> &source) : target(target), source(source)
	{
		c.resize(target.size());
		for (int x = 0; x < (int)target.size(); x++) c[x].resize(source.size());
#pragma omp parallel for
		for (int xc = 0; xc < (int)target.size(); xc++)
		{
			for (int yc = 0; yc < (int)source.size(); yc++)
			{
				c[xc][yc] = source.dist(yc, target, xc);
			}
		}
	}

	__forceinline__ AC getCostMulti(int x, int y) const { return c[x][y]; }
	__forceinline__ AC getCostMulti2(int x, int y) const { return c[x][getTargetMap(y)]; }
	__forceinline__ AC getCost(int x, int y) const { return c[getSourceMap(x)][getTargetMap(y)]; }
	__forceinline__ int getSourceCount(int x) const { return source.getCount(x); }
	__forceinline__ int getTargetCount(int y) const { return target.getCount(y); }
	__forceinline__ int getSourcePrefixSum(int x) const { return source.getPrefixSum(x); }
	__forceinline__ int getTargetPrefixSum(int y) const { return target.getPrefixSum(y); }
	__forceinline__ int getSourceSize() const { return (int)source.size(); }
	__forceinline__ int getTargetSize() const { return (int)target.size(); }
	__forceinline__ int getRealSourceSize() const { return source.realSize(); }
	__forceinline__ int getRealTargetSize() const { return target.realSize(); }
	__forceinline__ int getSourceMap(int x) const { return source.getMap(x); }
	__forceinline__ int getTargetMap(int y) const { return target.getMap(y); }

	template <bool PAR, class C>
	__forceinline__ void iterateMulti(const C &c, int x, AC &limit, std::vector<AC> &beta)
	{
		int yb = 0;
		if (PAR)
		{
#pragma omp parallel for
			for (int yy = 0; yy < (int)target.size(); yy++)
			{
				AC raw_cost = getCostMulti(x, yy);
				for (int yc = 0; yc < target.getCount(yy); yc++, yb++)
				{
					if (-beta[yy] <= limit)
					{
						AC ccost = raw_cost - beta[yb];
						c(yb, ccost);
					}
				}
			}
		}
		else
		{
			for (int yy = 0; yy < (int)target.size(); yy++)
			{
				AC raw_cost = getCostMulti(x, yy);
				for (int yc = 0; yc < target.getCount(yy); yc++, yb++)
				{
					if (-beta[yy] <= limit)
					{
						AC ccost = raw_cost - beta[yb];
						c(yb, ccost);
					}
				}
			}
		}
	}

	template <bool PAR, class C>
	__forceinline__ void iterateMulti(const C &c, int x, std::vector<AC> &beta)
	{
		int yb = 0;
		if (PAR)
		{
#pragma omp parallel for
			for (int yy = 0; yy < target.size(); yy++)
			{
				AC raw_cost = getCostMulti(x, yy);
				for (int yc = 0; yc < target.getCount(yy); yc++, yb++)
				{
					AC ccost = raw_cost - beta[yb];
					c(yb, ccost);
				}
			}
		}
		else
		{
			for (int yy = 0; yy < target.size(); yy++)
			{
				AC raw_cost = getCostMulti(x, yy);
				for (int yc = 0; yc < target.getCount(yy); yc++, yb++)
				{
					AC ccost = raw_cost - beta[yb];
					c(yb, ccost);
				}
			}
		}
	}

	template <bool PAR, class C>
	__forceinline__ void iterate(const C &c, int x, AC& limit, std::vector<AC> &beta)
	{
		iterateMulti<PAR>(c, getSourceMap(x), limit, beta);
	}
	template <bool PAR, class C>
	__forceinline__ void iterate(const C &c, int x, std::vector<AC> &beta)
	{
		iterateMulti<PAR>(c, getSourceMap(x), beta);
	}
	__forceinline__ void updateBeta(std::vector<AC> &beta, int r_count) {}
	__forceinline__ void printCases(bool reset) {}

	__forceinline__ bool cudaEnabled(int t) { return false; }
	__forceinline__ void cudaFill(std::vector<int> &idx, std::vector<AC> &heap, int x) {}
};
