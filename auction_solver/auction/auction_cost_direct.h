#pragma once

#include "auction_helper.h"

#include <omp.h>

template <class AC, typename GETCOST>
class DirectCost
{
private:
	int target_size;
	GETCOST getcost;

	std::vector<AC> *p_beta;
	std::vector<AC> *p_alpha;

public:
	DirectCost(GETCOST &getcost, int target_size)
		: target_size(target_size), getcost(getcost)
	{
	}

	~DirectCost()
	{
	}

	const AC getCost(int x, int y) const { return getcost(x, y); }

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
	void iterate(const C &c, int x,std::vector<AC> &beta)
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
	void iterateReverse(const C &c, int x, AC &limit, std::vector<AC> &alpha)
	{
		if (PAR)
		{
#pragma omp parallel for
			for (int yy = 0; yy < target_size; yy++)
			{
				if (-alpha[yy] <= limit)
				{
					AC ccost = getCost(yy, x) - alpha[yy];
					c(yy, ccost);
				}
			}
		}
		else
		{
			for (int yy = 0; yy < target_size; yy++)
			{
				if (-alpha[yy] <= limit)
				{
					AC ccost = getCost(yy, x) - alpha[yy];
					c(yy, ccost);
				}
			}
		}
	}

	template <bool PAR, class C>
	void iterateReverse(const C &c, int x, std::vector<AC> &alpha)
	{
		if (PAR)
		{
#pragma omp parallel for
			for (int yy = 0; yy < target_size; yy++)
			{
				AC ccost = getCost(yy, x) - alpha[yy];
				c(yy, ccost);
			}
		}
		else
		{
			for (int yy = 0; yy < target_size; yy++)
			{
				AC ccost = getCost(yy, x) - alpha[yy];
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

