#include "auction_cpu.h"

#include <thread>
#include <mutex>
#include <chrono>
#include <iostream>
#include <algorithm>

#include "auction_helper.h"

#include "auction_cost_table.h"
#include "auction_cost_direct.h"
#include "auction_cost_shielded.h"
#include "auction_cost_tree.h"

#include "auction_find_linear.h"
#include "auction_find_caching.h"

#include "auction_one_way.h"

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

const int epsilon_steps = 11;
static float epsilon_step[11] = { 100.0f, 10.0f, 1.0f, 0.1f, 0.01f, 0.001f, 0.0001f, 0.00001f, 0.000001f, 0.0000001f, 0.0f };
static float fraction_unmatched[11] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

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

auto global_l = []() { return -1.0f; };

template <class COST, class FIND, class L = decltype(global_l)>
void auctionSingle(std::vector<int> &coupling, COST &c, FIND &f, std::vector<AuctionCost> &beta, bool exact = false, int initial_epsilon = 0, L l = global_l)
{
	auto start_time = std::chrono::high_resolution_clock::now();

	int target_size = (int)coupling.size();
	std::vector<std::pair<volatile int, volatile AuctionCost>> B(target_size);
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

	//AuctionCost epsilon = initial_epsilon;
	int epsilon_idx = initial_epsilon;
	//if (exact) epsilon = 0.0f;
	if (exact) epsilon_idx = epsilon_steps - 1;

	int completed = 0;

#pragma omp parallel for
	for (int y = 0; y < target_size; y++)
	{
		B[y].first = -1;
		B[y].second = MAX_COST;
	}

#ifdef REUSE_BETA
	if (beta.size() == 0)
	{
		beta.resize(target_size);
#pragma omp parallel for
		for (int y = 0; y < target_size; y++)
		{
			beta[y] = AuctionCost(0.0);
		}
		c.updateBeta(beta, target_size);
		full_runs--;
	}
	else if (full_runs == 0)
	{
		//epsilon = INITIAL_EPSILON * (EPSILON_SCALE * EPSILON_SCALE * EPSILON_SCALE);
		epsilon_idx = epsilon_steps >> 2;
	}
	else
	{
		full_runs--;
	}
#else
	if (beta.size() == 0)
	{
		beta.resize(target_size);
	}
#pragma omp parallel for
	for (int y = 0; y < target_size; y++)
	{
		beta[y] = AuctionCost(0.0);
	}
	c.updateBeta(beta, target_size);
#endif

	std::vector<std::mutex> Block(B.size());
	// loop until done

	int elapsed = 0;
	//AuctionCost old_epsilon = epsilon * AuctionCost(2.0) + AuctionCost(1.0);
	AuctionCost old_epsilon = epsilon_step[epsilon_idx] * AuctionCost(2.0) + AuctionCost(1.0);
	int iteration = 0;

	AuctionOneWay<AuctionCost> auction;

	// check max cuda threads
	int num_cuda = 0;
	for (int i = 0; i < processor_count; i++) if (c.cudaEnabled(i)) num_cuda = i + 1;

	while (epsilon_idx < epsilon_steps)
	{
		AuctionCost epsilon = epsilon_step[epsilon_idx];
		int incomplete_allowed = (int)floor(target_size * fraction_unmatched[epsilon_idx]);
		epsilon_idx++;
		//f.updateAll(c, beta);
		completed = 0;
		// initialize coupling
		for (int i = 0; i < target_size; i++)
		{
			coupling[i] = -1;
			unassigned[0][i] = i;
		}
		u_count[0] = target_size;

		while (completed + incomplete_allowed < target_size)
		{
			displayProgress(start_time, elapsed, completed, target_size, iteration, epsilon, old_epsilon);
			iteration++;

			bool update = true;// ((((iteration ^ (iteration - 1)) + 1) >> 1) == iteration);

			//int old_completed = completed;

			if (epsilon != 0.0f)
			{
#ifdef DEFER_MISS
				completed = auction.auction<false>(coupling, beta, c, f, u_count, unassigned, d_count, deferred, r_count, receiving, B, Block, target_size, epsilon, update, num_cuda);
#else
				completed = auction.auction<false>(coupling, beta, c, f, u_count, unassigned, r_count, receiving, B, Block, target_size, epsilon, update);
#endif
			}
			else
			{
#ifdef DEFER_MISS
				completed = auction.auction<true>(coupling, beta, c, f, u_count, unassigned, d_count, deferred, r_count, receiving, B, Block, target_size, epsilon, update, num_cuda);
#else
				completed = auction.auction<true>(coupling, beta, c, f, u_count, unassigned, r_count, receiving, B, Block, target_size, epsilon, update);
#endif
			}

			if (completed + incomplete_allowed >= target_size)
			{
				if (completed == target_size)
				{
					float cost = l();
					if (cost >= 0.0f) displayProgressCost(start_time, elapsed, completed, target_size, iteration, epsilon, cost);
					else displayProgress(start_time, elapsed, completed, target_size, iteration, epsilon, old_epsilon, true);
				}
				else
				{
					displayProgress(start_time, elapsed, completed, target_size, iteration, epsilon, old_epsilon, true);
				}
			}
		}
		// find maximum beta (for stability)
		AuctionCost max_beta = beta[0];
		for (int y = 1; y < target_size; y++) max_beta = std::max(max_beta, beta[y]);
		for (int y = 0; y < target_size; y++) beta[y] -= max_beta;
		f.fixBeta(max_beta);
		c.updateBeta(beta, target_size);
	}
}

template <class COST, class FIND>
void assignSingle(std::vector<int> &coupling, COST &c, FIND &f, std::vector<AuctionCost> &beta)
{
	//auto start_time = std::chrono::high_resolution_clock::now();

	int target_size = (int)coupling.size();
	std::vector<std::vector<int>> unassigned(2);
	unassigned[0].resize(target_size);
	unassigned[1].resize(target_size);

	std::vector<int> u_count(2);
	u_count[0] = u_count[1] = 0;

#ifdef DEFER_MISS
	std::vector<int> deferred(target_size);
	int d_count = 0;
#endif

	int completed = 0;

#ifdef REUSE_BETA
	if (beta.size() == 0)
	{
		beta.resize(target_size);
#pragma omp parallel for
		for (int y = 0; y < target_size; y++)
		{
			beta[y] = AuctionCost(0.0);
		}
		c.updateBeta(beta, target_size);
	}
#else
	if (beta.size() == 0)
	{
		beta.resize(target_size);
	}
#pragma omp parallel for
	for (int y = 0; y < target_size; y++)
	{
		beta[y] = AuctionCost(0.0);
	}
	c.updateBeta(beta, target_size);
#endif

	//int elapsed = 0;

	AuctionOneWay<AuctionCost> auction;

	// check max cuda threads
	int num_cuda = 0;
	for (int i = 0; i < processor_count; i++) if (c.cudaEnabled(i)) num_cuda = i + 1;

	{
		// initialize coupling
		for (int i = 0; i < target_size; i++)
		{
			coupling[i] = -1;
			unassigned[0][i] = i;
		}
		u_count[0] = target_size;

		{
			bool update = true;// ((((iteration ^ (iteration - 1)) + 1) >> 1) == iteration);

			//int old_completed = completed;

#ifdef DEFER_MISS
			completed = auction.assign(coupling, beta, c, f, u_count, unassigned, d_count, deferred, target_size, update, num_cuda);
#else
			completed = auction.assign(coupling, beta, c, f, u_count, unassigned, target_size, update);
#endif
		}
#if 1
		// find maximum beta (for stability)
		AuctionCost max_beta = beta[0];
		for (int y = 1; y < target_size; y++) max_beta = std::max(max_beta, beta[y]);
		for (int y = 0; y < target_size; y++) beta[y] -= max_beta;
		f.fixBeta(max_beta);
		c.updateBeta(beta, target_size);
#endif
	}
}

template <class COST, class FIND>
void auctionMultiple(std::vector<int> &coupling, COST &c, FIND &f, std::vector<AuctionCost> &beta, AuctionCost initial_epsilon, AuctionCost min_epsilon)
{
	auto start_time = std::chrono::high_resolution_clock::now();

	std::vector<std::pair<volatile int, volatile AuctionCost>> B(c.getRealTargetSize());

	std::vector<std::vector<int>> unassigned(2);
	unassigned[0].resize(c.getSourceSize());
	unassigned[1].resize(c.getSourceSize());
	std::vector<std::vector<int>> unassigned_count(2);
	unassigned_count[0].resize(c.getSourceSize());
	unassigned_count[1].resize(c.getSourceSize());
	std::vector<int> receiving(c.getRealTargetSize());

	int r_count;
	std::vector<int> u_count(2);

	AuctionCost epsilon = initial_epsilon;

	int completed = 0;

	std::vector<std::mutex> Block(B.size());
	// loop until done

	int elapsed = 0;
	AuctionCost old_epsilon = epsilon * AuctionCost(2.0);
	int iteration = 0;

	AuctionOneWay<AuctionCost> auction;

	// force correct solution
	//epsilon = min_epsilon;

	int epsilon_idx = 0;

	while (epsilon_idx + 1 < epsilon_steps)
	{
		epsilon = epsilon_step[epsilon_idx++];
		u_count[0] = u_count[1] = 0;
//		u_count = 0;
		r_count = 0;

#pragma omp parallel for
		for (int y = 0; y < c.getRealTargetSize(); y++)
		{
			beta[y] = AuctionCost(0.0);
			B[y].first = -1;
			B[y].second = MAX_COST;
		}

		//f.updateAll(c, beta);
		completed = 0;
		// initialize coupling
		for (int i = 0; i < c.getSourceSize(); i++)
		{
#if 1
			unassigned[0][i] = i;
			unassigned_count[0][i] = c.getSourceCount(i);
			unassigned_count[1][i] = 0;
#else
			unassigned[i] = i;
			unassigned_count[i] = c.getSourceCount(i);
#endif
		}
		for (int i = 0; i < c.getRealTargetSize(); i++)
		{
			coupling[i] = -1;
		}
		u_count[0] = c.getSourceSize();
//		u_count = c.getSourceSize();

		while (completed < c.getRealTargetSize())
		{
			displayProgress(start_time, elapsed, completed, c.getRealTargetSize(), iteration, epsilon, old_epsilon);
			iteration++;

			completed = auction.auctionMultiple(coupling, beta, c, f, u_count, unassigned, unassigned_count, r_count, receiving, B, Block, epsilon);

			if (completed == c.getRealTargetSize()) displayProgress(start_time, elapsed, completed, c.getRealTargetSize(), iteration, epsilon, old_epsilon, true);
		}
#if 1
		// find maximum beta (for stability)
		AuctionCost max_beta = beta[0];
		for (int y = 1; y < c.getRealTargetSize(); y++) max_beta = std::max(max_beta, beta[y]);
		for (int y = 0; y < c.getRealTargetSize(); y++) beta[y] -= max_beta;
		f.fixBeta(max_beta);
#endif
		
		//epsilon *= EPSILON_SCALE;
		//if (epsilon <= 0.99f * min_epsilon) epsilon = -1.0f;
//		if (epsilon <= 1.01f * MIN_EPSILON) epsilon = -1.0f;
//		else epsilon *= EPSILON_SCALE;
	}
}

bool auctionAlgorithmTryMulti(int *linear_index, float *linear_target, int target_size, float *linear_source, int source_size, int total_channel)
{
	CostMultiple<AuctionCost> multiple_target(linear_target, target_size, target_size, total_channel);
	CostMultiple<AuctionCost> multiple_source(linear_source, source_size, target_size, total_channel);

	size_t total_input = (size_t)target_size * (size_t)target_size;
	size_t total_reduced = multiple_source.size() * multiple_target.size();

	bool final_regular = false;
	if (total_reduced / 10 >= total_input / 11)
	{
		final_regular = true;
	}

	int reduction = 2;

	std::vector<int> coupling(target_size);
	{
		std::vector<AuctionCost> beta(target_size);

		bool complete = false;
		AuctionCost last_epsilon = epsilon_step[0];
		int epsilon_idx = 0;
		while (!complete)
		{
			CostMultiple<AuctionCost> multiple_target(linear_target, target_size, target_size, total_channel, reduction, linear_source, source_size);
			CostMultiple<AuctionCost> multiple_source(linear_source, source_size, target_size, total_channel, reduction, linear_target, target_size);
			size_t total_reduced_current = multiple_source.size() * multiple_target.size();
			AuctionCost min_epsilon = multiple_target.getEpsilon();// std::max(AuctionCost(0.1), MIN_EPSILON * (AuctionCost)(10) * (AuctionCost)total_reduced / (AuctionCost)total_reduced_current);
			//if (last_epsilon == epsilon_step[0]) last_epsilon = min_epsilon * 10000.0f;
			//else last_epsilon = min_epsilon * 10.0f;
			if ((total_reduced_current << 2) > total_reduced)
			{
				complete = true;
			}
			else
			{
				reduction <<= 1;
				if (total_reduced_current <= 1024 * 1024)
				{
					//if (swapped) std::swap(multiple_source, multiple_target);
					TableCostMultiple<AuctionCost> c(multiple_target, multiple_source);
					FindLinearMultiple<TableCostMultiple<AuctionCost>, AuctionCost> f(target_size);
					std::cout << "min_epsilon = " << min_epsilon << " (linear table)" << std::endl;
					auctionMultiple(coupling, c, f, beta, last_epsilon, min_epsilon);
				}
				else if (total_reduced_current <= 32768 * 32768)
				{
					TableCostMultiple<AuctionCost> c(multiple_target, multiple_source);
					FindLinearMultiple<TableCostMultiple<AuctionCost>, AuctionCost> f(target_size);
//					FindCachingMultiple<TableCostMultiple<AuctionCost>, CACHE1, AuctionCost> f(target_size);
					std::cout << "min_epsilon = " << min_epsilon << " (caching table)" << std::endl;
					auctionMultiple(coupling, c, f, beta, last_epsilon, min_epsilon);
				}
				else
				{
					DirectCostMultiple<AuctionCost> c(multiple_target, multiple_source);
					FindLinearMultiple<DirectCostMultiple<AuctionCost>, AuctionCost> f(target_size);
//					FindCachingMultiple<DirectCostMultiple<AuctionCost>, CACHE2, AuctionCost> f(target_size);
					std::cout << "min_epsilon = " << min_epsilon << " (caching direct)" << std::endl;
					auctionMultiple(coupling, c, f, beta, last_epsilon, min_epsilon);
				}
				epsilon_idx++;
				if (epsilon_idx + 1 == epsilon_steps) complete = true;
				else last_epsilon = epsilon_step[epsilon_idx];
			}
		}

		if (final_regular)
		{
			if (target_size <= 1024)
			{
				TableCost<AuctionCost> c(linear_target, target_size, linear_source, source_size, total_channel);
				FindLinear<TableCost<AuctionCost>, AuctionCost> f(target_size);
				std::cout << "regular linear table" << std::endl;
				auctionSingle(coupling, c, f, beta, false, epsilon_steps - 1);
			}
			else if (target_size <= 32768)
			{
				TableCost<AuctionCost> c(linear_target, target_size, linear_source, source_size, total_channel);
				FindCaching<TableCost<AuctionCost>, CACHE1, AuctionCost> f(target_size, c, beta);
				std::cout << "regular caching table" << std::endl;
				auctionSingle(coupling, c, f, beta, false, epsilon_steps - 1);
			}
			else
			{
				DirectCost<AuctionCost> c(linear_target, target_size, linear_source, source_size, total_channel);
				FindCaching<DirectCost<AuctionCost>, CACHE2, AuctionCost> f(target_size, c, beta);
				std::cout << "regular caching direct" << std::endl;
				auctionSingle(coupling, c, f, beta, false, epsilon_steps - 1);
			}
		}
		else
		{
			if (total_reduced <= 1024 * 1024)
			{
				TableCostMultiple<AuctionCost> c(multiple_target, multiple_source);
				{
					FindLinearMultiple<TableCostMultiple<AuctionCost>, AuctionCost> f(target_size);
					std::cout << "multi linear table" << std::endl;
					auctionMultiple(coupling, c, f, beta, last_epsilon, MIN_EPSILON);
				}
				{
					FindLinearMultiple<TableCostMultiple<AuctionCost>, AuctionCost> f(target_size);
					std::cout << "regular linear table" << std::endl;
					auctionSingle(coupling, c, f, beta, true, epsilon_steps - 1);
				}
			}
			else if (total_reduced <= 32768 * 32768)
			{
				TableCostMultiple<AuctionCost> c(multiple_target, multiple_source);
				{
					FindCachingMultiple<TableCostMultiple<AuctionCost>, CACHE1, AuctionCost> f(target_size, c, beta);
					auctionMultiple(coupling, c, f, beta, last_epsilon, MIN_EPSILON);
					std::cout << "multi caching table" << std::endl;
				}
				{
					FindCachingMultiple<TableCostMultiple<AuctionCost>, CACHE1, AuctionCost> f(target_size, c, beta);
					std::cout << "regular caching table" << std::endl;
					auctionSingle(coupling, c, f, beta, true, epsilon_steps - 1);
				}
			}
			else
			{
				DirectCostMultiple<AuctionCost> c(multiple_target, multiple_source);
				{
					FindCachingMultiple<DirectCostMultiple<AuctionCost>, CACHE2, AuctionCost> f(target_size, c, beta);
					std::cout << "multi caching direct" << std::endl;
					auctionMultiple(coupling, c, f, beta, last_epsilon, MIN_EPSILON);
				}
				{
					FindCachingMultiple<DirectCostMultiple<AuctionCost>, CACHE2, AuctionCost> f(target_size, c, beta);
					std::cout << "regular caching direct" << std::endl;
					auctionSingle(coupling, c, f, beta, true, epsilon_steps - 1);
				}
			}
		}
	}

	multiple_source.unmap(coupling);
	// unmapUnique on indices...
	std::vector<int> indirect(coupling.size());
	for (int i = 0; i < (int)indirect.size(); i++) indirect[i] = i;
	multiple_target.unmapUnique(indirect);

	{
#pragma omp parallel for
		for (int x = 0; x < target_size; x++)
		{
			linear_index[indirect[x]] = coupling[x];
		}
	}

	return true;
}

DirectCost<AuctionCost> *g_c = NULL;
FindCaching<DirectCost<AuctionCost>, CACHE2, AuctionCost> *g_f = NULL;

void auctionAlgorithmCleanup()
{
	if (g_c != NULL)
	{
		delete g_c;
		g_c = NULL;
	}
	if (g_f != NULL)
	{
		delete g_f;
		g_f = NULL;
	}
}

void auctionAlgorithm(int *linear_index, float *linear_target, int target_size, float *linear_source, int source_size, int total_channel, std::vector<AuctionCost> &beta, bool relaxed)
{
	processor_count = omp_get_num_procs();
	omp_set_num_threads(processor_count);
	// potentially running cuda with multiple threads
	int device;
	cudaGetDevice(&device);

	std::cout << "maximum number of threads: " << omp_get_max_threads() << std::endl;

#ifndef REUSE_BETA
#pragma omp parallel for
	for (int x = 0; x < beta.size(); x++) beta[x] = AuctionCost(0.0);
#endif

	auto start_time = std::chrono::high_resolution_clock::now();
	// non-reduced
	std::vector<int> coupling(target_size);

	if (target_size <= 1024)
	{
		TableCost<AuctionCost> c(linear_target, target_size, linear_source, source_size, total_channel);
		FindLinear<TableCost<AuctionCost>, AuctionCost> f(target_size);
		displayTime(start_time, "setup completed");
		if (relaxed)
			assignSingle(coupling, c, f, beta);
		else
			auctionSingle(coupling, c, f, beta, false, 0, [&]()
				{
#pragma omp parallel for
					for (int x = 0; x < target_size; x++)
					{
						// slack...
						if (coupling[x] != -1) linear_index[coupling[x]] = x;
					}
					float cost = getCurrentCost(linear_index, linear_target, target_size, linear_source, source_size, total_channel) * total_channel;
					return cost;
				});
	}
	else
	{
		if (g_c == NULL)
		{
			g_c = new DirectCost<AuctionCost>(linear_target, target_size, linear_source, source_size, total_channel);
			g_f = new FindCaching<DirectCost<AuctionCost>, CACHE2, AuctionCost>(target_size, *g_c, beta);
		}
		else
		{
#ifdef REUSE_CACHE
			g_c->updateBeta(beta, target_size);
			g_f->recreate(*g_c, linear_target);
#else
			delete g_f;
			delete g_c;
			g_c = new DirectCost<AuctionCost>(linear_target, target_size, linear_source, source_size, total_channel);
			g_f = new FindCaching<DirectCost<AuctionCost>, CACHE2, AuctionCost>(target_size, *g_c, beta);
#endif
		}
		DirectCost<AuctionCost> &c = *g_c;
		FindCaching<DirectCost<AuctionCost>, CACHE2, AuctionCost> &f = *g_f;
		displayTime(start_time, "setup completed");
		if (relaxed)
			assignSingle(coupling, c, f, beta);
		else
			auctionSingle(coupling, c, f, beta, false, 0, [&]()
				{
#pragma omp parallel for
					for (int x = 0; x < target_size; x++)
					{
						// slack...
						if (coupling[x] != -1) linear_index[coupling[x]] = x;
					}
					float cost = getCurrentCost(linear_index, linear_target, target_size, linear_source, source_size, total_channel) * total_channel;
					return cost;
				});
	}

	if (relaxed)
	{
		// coupling is the other way round
#pragma omp parallel for
		for (int x = 0; x < target_size; x++)
		{
			linear_index[x] = coupling[x];
		}
	}
	else
	{
#pragma omp parallel for
		for (int x = 0; x < target_size; x++)
		{
			// slack...
			if (coupling[x] != -1) linear_index[coupling[x]] = x;
		}
	}

	cudaSetDevice(device);
}

AuctionCost getCurrentCost(int *linear_index, float *linear_target, int target_size, float *linear_source, int source_size, int total_channel)
{
	AuctionCost cost = AuctionCost(0.0);
	{
		DirectCost<AuctionCost> c(linear_target, target_size, linear_source, source_size, total_channel);
		for (int i = 0; i < target_size; i++)
		{
			cost += c.getCost(i, linear_index[i]);
		}
	}
	return cost / (AuctionCost)target_size;
}
