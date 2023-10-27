#pragma once

#include <vector>
#include "auction_helper.h"

template <class AC>
class TableCost
{
private:
	std::vector<std::vector<AC>> c;
	int target_size;
public:
	template <typename GETCOST>
	TableCost(GETCOST &getcost, int target_size)
		: target_size(target_size)
	{
		c.resize(target_size);
		for (int x = 0; x < target_size; x++) c[x].resize(target_size);
#pragma omp parallel for
		for (int x = 0; x < target_size; x++)
		{
			for (int y = 0; y < target_size; y++)
			{
				c[x][y] = getcost(x, y);
			}
		}
	}
	const AC getCost(int x, int y) const { return c[x][y]; }

	template <bool PAR, class C>
	void iterate(const C &c, int x, AC &limit, std::vector<AC> &beta)
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
	void iterate(const C &c, int x, std::vector<AC> &beta)
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
	void iterateReverse(const C &c, int x, AC &limit, std::vector<AC> &beta)
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
	void iterateReverse(const C &c, int x, std::vector<AC> &beta)
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
	void updateBeta(std::vector<AC> &beta, int r_count) {}
	void updateAlpha(std::vector<AC> &alpha, int r_count) {}
	void printCases(bool reset) {}
	bool requiresCaching() { return false; }
};
