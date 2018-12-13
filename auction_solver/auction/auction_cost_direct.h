#pragma once

#include "auction_helper.h"

#include <omp.h>

template <class AC>
class DirectCost
{
private:
	int target_size;
	int total_channel;
	float *own_target;
	float *own_source;

	std::vector<AC> *p_beta;
	std::vector<AC> *p_alpha;

	int device_count;

public:
	DirectCost(float *linear_target, int target_size, float *linear_source, int source_size, int total_channel)
		: target_size(target_size), total_channel(total_channel)
	{
		own_target = new float[target_size * total_channel];
		own_source = new float[target_size * total_channel];
#pragma omp parallel for
		for (int x = 0; x < target_size; x++)
		{
			//sorted[x] = x;
			int x0 = src_idx(x, source_size, target_size);
			for (int cc = 0; cc < total_channel; cc++)
			{
				own_source[x * total_channel + cc] = linear_source[cc * source_size + x0];
				own_target[x * total_channel + cc] = linear_target[cc * target_size + x];
			}
		}
	}

	float update_target(float *linear_target, int x)
	{
		float mod = 0.0f;
		for (int cc = 0; cc < total_channel; cc++)
		{
			float d = own_target[x * total_channel + cc] - linear_target[cc * target_size + x];
			mod += d * d;
			own_target[x * total_channel + cc] = linear_target[cc * target_size + x];
		}
		return mod / (float)total_channel;
	}

	~DirectCost()
	{
		delete[] own_target;
		delete[] own_source;
	}

	const AC getCost(int x, int y) const
	{
		AC cost = AC(0.0);
		int sidx = y * total_channel;
		int tidx = x * total_channel;
		for (int cc = 0; cc < total_channel; cc++)
		{
			AC d = own_source[sidx++] - own_target[tidx++];
			cost = cost + d * d;
		}
		cost /= (AC)total_channel;
		return cost;
	}

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

