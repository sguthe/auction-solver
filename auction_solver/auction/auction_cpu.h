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

#include "../../lap_solver/core/lap_solver.h"

#include <omp.h>

// currently broken
#define REUSE_CACHE
#define REUSE_BETA

//#define MAX_COST 1.0e80
//#define EPSILON_SCALE 0.2
//#define MIN_EPSILON 0.0002
//#define INITIAL_EPSILON 1953.125

//#define EPSILON_SCALE 0.1f
//#define INITIAL_EPSILON 100.0f
//#define INITIAL_EPSILON 976.5625f

//const int epsilon_steps = 8;
//static float epsilon_step[8] = { 100.0f, 10.0f, 1.0f, 0.1f, 0.01f, 0.001f, 0.0001f, 0.0f };
//static float fraction_unmatched[8] = { 0.006f, 0.005f, 0.004f, 0.003f, 0.002f, 0.001f, 0.0f, 0.0f };
//static float fraction_unmatched[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

//const int epsilon_steps = 26;
//static float epsilon_step[26] = { 1600.0f, 800.0f, 400.0f, 200.0f, 100.0f, 50.0f, 25.0f, 12.6f, 6.3f, 3.2f, 1.6f, 0.8f, 0.4f, 0.2f, 0.1f, 0.05f, 0.025f, 0.0126f, 0.0063f, 0.0032f, 0.0016f, 0.0008f, 0.0004f, 0.0002f, 0.0001f, 0.0f };
//static float fraction_unmatched[26] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

//const int epsilon_steps = 8;
//static float epsilon_step[8] = { 1.0f, 0.1f, 0.01f, 0.001f, 0.0001f, 0.00001f, 0.000001f, 0.0f };
//static float fraction_unmatched[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

//const int epsilon_steps = 11;
//static float epsilon_step[11] = { 1.0f, 0.1f, 0.01f, 0.001f, 0.0001f, 0.00001f, 0.000001f, 0.0000001f, 0.00000001f, 0.000000001f, 0.0f };
//static float fraction_unmatched[11] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

#define CACHE1 127
#define CACHE2 4095
//#define CACHE2 2047

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

static int full_runs = 3;

template <class AC, class COST, class FIND, class L>
AC auctionSingle(std::vector<int> &coupling, COST &c, FIND &f, std::vector<AC> &beta, AC initial_epsilon, L l, bool parallel)
{
	auto start_time = std::chrono::high_resolution_clock::now();

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

	int elapsed = 0;
	AC epsilon_lower = epsilon / AC(25 * target_size);
	epsilon *= AC(25);
	AC old_epsilon = epsilon * AC(2) + AC(1);
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

			bool update = true;

			if (epsilon != 0.0f)
			{
#ifdef DEFER_MISS
				completed = auction.auction<false>(coupling, beta, c, f, u_count, unassigned, d_count, deferred, r_count, receiving, B, Block, target_size, epsilon, parallel);
#else
				completed = auction.auction<false>(coupling, beta, c, f, u_count, unassigned, r_count, receiving, B, Block, target_size, epsilon, parallel);
#endif
			}
			else
			{
#ifdef DEFER_MISS
				completed = auction.auction<true>(coupling, beta, c, f, u_count, unassigned, d_count, deferred, r_count, receiving, B, Block, target_size, epsilon, parallel);
#else
				completed = auction.auction<true>(coupling, beta, c, f, u_count, unassigned, r_count, receiving, B, Block, target_size, epsilon, parallel);
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

#if 0
template <class AC, class GETCOST>
void auctionAlgorithm(int *linear_index, GETCOST &getcost, int target_size, AC initial_epsilon, bool table, bool caching, bool parallel, int max_size)
{
	processor_count = omp_get_num_procs();
	omp_set_num_threads(processor_count);

	std::cout << "Auction solver using " << omp_get_max_threads() << " threads." << std::endl;

	std::vector<AC> beta;

	auto start_time = std::chrono::high_resolution_clock::now();
	// non-reduced
	std::vector<int> coupling(target_size);

	int cache_size = (int)ceil(sqrt((double)target_size));

	if (table)
	{
		TableCost<AC> c(getcost, target_size);
		FindCaching<TableCost<AC>, AC> f(target_size, c, beta, cache_size);
		lap::displayTime(start_time, "setup completed", std::cout);
		auctionSingle(coupling, c, f, beta, initial_epsilon, [&]()
		{
#pragma omp parallel for
			for (int x = 0; x < target_size; x++)
			{
				// slack...
				if (coupling[x] != -1) linear_index[coupling[x]] = x;
			}
			AC cost = getCurrentCost<AC, GETCOST>(linear_index, getcost, target_size);
			return cost;
		}, parallel);
	}
	else
	{
		DirectCost<AC, GETCOST> c(getcost, target_size);
		FindCaching<DirectCost<AC, GETCOST>, AC> f(target_size, c, beta, cache_size);
		lap::displayTime(start_time, "setup completed", std::cout);
		auctionSingle(coupling, c, f, beta, initial_epsilon, [&]()
		{
#pragma omp parallel for
			for (int x = 0; x < target_size; x++)
			{
				// slack...
				if (coupling[x] != -1) linear_index[coupling[x]] = x;
			}
			AC cost = getCurrentCost<AC, GETCOST>(linear_index, getcost, target_size);
			return cost;
		}, parallel);
	}

#pragma omp parallel for
	for (int x = 0; x < target_size; x++)
	{
		// slack...
		if (coupling[x] != -1) linear_index[coupling[x]] = x;
	}
}
#endif

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
