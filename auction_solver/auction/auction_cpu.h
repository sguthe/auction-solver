#pragma once

#include <vector>

#include <thread>
#include <mutex>
#include <chrono>
#include <iostream>
#include <algorithm>

#include "auction_helper.h"

#include "auction_cost_table.h"
#include "auction_cost_direct.h"
#include "auction_cost_adapter.h"

#include "auction_find_linear.h"
#include "auction_find_caching.h"

#include "auction_one_way.h"

#include <omp.h>

#ifdef DISPLAY_MISS
long long hit_count = 0;
long long miss_count = 0;

long long total_hit_count = 0;
long long total_miss_count = 0;
#endif

#ifdef DISPLAY_THREAD_FILL
std::vector<long long> thread_fill_count;
#endif

int processor_count;

template <class AC, class COST, class FIND, class L>
AC auctionSingle(std::vector<int> &coupling, COST &c, FIND &f, std::vector<AC> &beta, AC initial_epsilon, L l, bool parallel)
{
#ifndef LAP_QUIET
	auto start_time = std::chrono::high_resolution_clock::now();
#endif

	int target_size = (int)coupling.size();
	std::vector<std::pair<volatile int, volatile AC>> B(target_size);
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

	AC epsilon = initial_epsilon;

	int completed = 0;

	if (parallel)
	{
#pragma omp parallel for
		for (int y = 0; y < target_size; y++)
		{
			B[y].first = -1;
			B[y].second = MAX_COST;
		}
	}
	else
	{
		for (int y = 0; y < target_size; y++)
		{
			B[y].first = -1;
			B[y].second = MAX_COST;
		}
	}

	if (beta.size() == 0)
	{
		beta.resize(target_size);
	}

	if (parallel)
	{
#pragma omp parallel for
		for (int y = 0; y < target_size; y++)
		{
			beta[y] = AC(0);
		}
	}
	else
	{
		for (int y = 0; y < target_size; y++)
		{
			beta[y] = AC(0);
		}
	}
	c.updateBeta(beta, target_size);

	std::vector<std::mutex> Block(B.size());
	// loop until done

#ifndef LAP_QUIET
	int elapsed = 0;
#endif
	AC epsilon_lower = epsilon / AC(25 * target_size);
	epsilon *= AC(25);
	int iteration = 0;

	AuctionOneWay<AC> auction;

	bool first = true;

	AC max_beta(0);
	AC cost(0);

	while (epsilon >= AC(0))
	{
		completed = 0;

		if (epsilon > AC(0))
		{
			if (!first)
			{
				epsilon *= AC(0.2);
				if (epsilon < epsilon_lower) epsilon = AC(0);
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
				completed = auction.template auction<false>(coupling, beta, c, f, u_count, unassigned, d_count, deferred, r_count, receiving, B, Block, target_size, epsilon, parallel);
#else
				completed = auction.template auction<false>(coupling, beta, c, f, u_count, unassigned, r_count, receiving, B, Block, target_size, epsilon, parallel);
#endif
			}
			else
			{
#ifdef DEFER_MISS
				completed = auction.template auction<true>(coupling, beta, c, f, u_count, unassigned, d_count, deferred, r_count, receiving, B, Block, target_size, epsilon, parallel);
#else
				completed = auction.template auction<true>(coupling, beta, c, f, u_count, unassigned, r_count, receiving, B, Block, target_size, epsilon, parallel);
#endif
			}

#ifndef LAP_QUIET
			if (completed == target_size)
			{
				cost = l();
				std::stringstream ss;
				ss << " epsilon = " << epsilon;
				if (cost > AC(0)) ss << " cost = " << cost;
				lap::displayProgress(start_time, elapsed, completed, target_size, ss.str().c_str(), iteration);
			}
#endif
		}

		// check for epsilon == 0 and ensure termination
		if (epsilon == AC(0)) epsilon = AC(-1);
		else
		{
			// find maximum beta (for stability)
			max_beta = beta[0];
			for (int y = 1; y < target_size; y++) max_beta = std::max(max_beta, beta[y]);
			for (int y = 0; y < target_size; y++) beta[y] -= max_beta;
			f.fixBeta(max_beta);
			c.updateBeta(beta, target_size);
		}
	}
#ifdef LAP_QUIET
	cost = l();
#endif
	return cost;
}

template <class AC, class GETCOST>
AC getCurrentCost(int *linear_index, GETCOST &c, int target_size)
{
	AC cost = AC(0);
	{
		for (int i = 0; i < target_size; i++)
		{
			cost += c.getCost(i, linear_index[i]);
		}
	}
	return cost;
}
