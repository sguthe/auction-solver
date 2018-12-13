#pragma once

#include "auction_helper.h"
#include <mutex>
#include <omp.h>

extern int processor_count;

#define BUYING_THREAD_COUNT 1
#define BIDDING_THREAD_COUNT 1
//#define BUYING_THREAD_COUNT (processor_count >> 2)
//#define BIDDING_THREAD_COUNT (processor_count >> 2)

template <class AC>
class AuctionOneWay
{
private:
	std::vector<std::vector<int>> m_y;
	std::vector<std::vector<AC>> m_cost;
	int m_real_pending;
private:
	void enqueue(int v, int &c, std::vector<int> &l)
	{
		l[c++] = v;
	}
	void enqueue(int v, int &c, std::vector<int> &l, std::vector<int> &u)
	{
		if (u[v] == 0) l[c++] = v;
		u[v]++;
	}

	template <bool PAR>
	void auctionBidding(
		int x, int y, std::pair<AC, AC> cost, 
		std::vector<int> &u_count, std::vector<std::vector<int>> &unassigned, 
		int &r_count, std::vector<int> &receiving, 
		std::vector<std::pair<volatile int, volatile AC>> &B, std::vector<std::mutex> &Block)
	{
		AC ccost = cost.first - cost.second;
		AC first_cost = B[y].second;
		int first_index = B[y].first;
		int queue_index = -1;
		if (y == -1)
		{
			queue_index = x;
		}
		else
		{
			if ((ccost < first_cost) || ((ccost == first_cost) && (x < first_index)))
			{
				if (!PAR)
				{
					std::lock_guard<std::mutex> lock(Block[y]);
					AC second_cost = B[y].second;
					int second_index = B[y].first;
					if ((ccost < second_cost) || ((ccost == second_cost) && (x < second_index)))
					{
						if (second_index != -1) queue_index = second_index;
						else
						{
#pragma omp critical
							enqueue(y, r_count, receiving);
						}
						B[y].first = x;
						B[y].second = ccost;
					}
					else
					{
						queue_index = x;
					}
				}
				else
				{
					if (first_index != -1) queue_index = first_index;
					else enqueue(y, r_count, receiving);
					B[y].first = x;
					B[y].second = ccost;
				}
			}
			else
			{
				queue_index = x;
			}
		}
		if (!PAR)
		{
			if (queue_index != -1)
			{
#pragma omp critical
				enqueue(queue_index, u_count[0], unassigned[0]);
			}
		}
		else
		{
			if (queue_index != -1) enqueue(queue_index, u_count[0], unassigned[0]);
		}
	}

	template <bool EXACT>
	void auctionBuying(
		std::vector<int> &coupling, std::vector<AC> &beta, 
		std::vector<int> &u_count, std::vector<std::vector<int>> &unassigned, 
		int &r_count, std::vector<int> &receiving, 
		std::vector<std::pair<volatile int, volatile AC>> &B, AC epsilon)
	{
		const int rc = r_count;
		bool updated = false;

		if (rc >= BUYING_THREAD_COUNT)
		{
#pragma omp parallel for schedule(static)
			for (int yi = 0; yi < rc; yi++)
			{
				int y = receiving[yi];
				int queue_index = -1;
				if ((!EXACT) || ((B[y].second < 0.0f) || ((B[y].second == 0.0f) && ((coupling[y] == -1) || (B[y].first < coupling[y])))))
				{
					updated = true;
					if (coupling[y] != -1)
					{
						queue_index = coupling[y];
						coupling[y] = -1;
					}
					AC cost = B[y].second;
					int x = B[y].first;

					if (EXACT) beta[y] += cost;
					else beta[y] = std::min(beta[y] + cost - epsilon, beta[y] * 1.00000012f);

					coupling[y] = x;
				}
				else
				{
					queue_index = B[y].first;
				}
				if (queue_index != -1)
				{
#pragma omp critical
					enqueue(queue_index, u_count[0], unassigned[0]);
				}
				B[y].first = -1;
				B[y].second = MAX_COST;
			}
		}
		else
		{
			for (int yi = 0; yi < rc; yi++)
			{
				int y = receiving[yi];
				int queue_index = -1;
				if ((!EXACT) || ((B[y].second < 0.0f) || ((B[y].second == 0.0f) && ((coupling[y] == -1) || (B[y].first < coupling[y])))))
				{
					updated = true;
					if (coupling[y] != -1)
					{
						queue_index = coupling[y];
						coupling[y] = -1;
					}
					AC cost = B[y].second;
					int x = B[y].first;

					if (EXACT) beta[y] += cost;
					else beta[y] = std::min(beta[y] + cost - epsilon, beta[y] * 1.00000012f);

					coupling[y] = x;
				}
				else
				{
					queue_index = B[y].first;
				}
				if (queue_index != -1)
				{
					enqueue(queue_index, u_count[0], unassigned[0]);
				}
				B[y].first = -1;
				B[y].second = MAX_COST;
			}
		}

		if ((EXACT) && (!updated))
		{
#pragma omp parallel for schedule(static)
			for (int yi = 0; yi < rc; yi++)
			{
				int y = receiving[yi];
 				beta[y] = std::min(beta[y] - 1e-9f, beta[y] * 1.00000012f);
			}
		}
		else if (!updated)
		{
			std::cout << "This should never happen!!" << std::endl;
			std::cout.flush();
		}

		r_count = 0;
	}

public:
	AuctionOneWay()
	{
		m_y.resize(omp_get_max_threads());
		m_cost.resize(omp_get_max_threads());
	}

	template <bool EXACT, class COST, class FIND>
	int auction(
		std::vector<int> &coupling, std::vector<AC> &beta, 
		COST &c, FIND &f, 
		std::vector<int> &u_count, std::vector<std::vector<int>> &unassigned, 
#ifdef DEFER_MISS
		int &d_count, std::vector<int> &deferred,
#endif
		int &r_count, std::vector<int> &receiving,
		std::vector<std::pair<volatile int, volatile AC>> &B, std::vector<std::mutex> &Block, 
		int target_size, AC epsilon, bool update)
	{
		std::swap(unassigned[0], unassigned[1]);
		u_count[1] = u_count[0];
		u_count[0] = 0;
		const int uc = u_count[1];

#ifdef DEFER_MISS
		d_count = 0;
#endif

		if (uc >= BIDDING_THREAD_COUNT)
		{
#ifdef DEFER_MISS
#pragma omp parallel for
#else
#pragma omp parallel for schedule(dynamic)
#endif
			for (int xi = 0; xi < uc; xi++)
			{
				int x = unassigned[1][xi];
				std::pair<int, int> y;
				std::pair<AC, AC> cost;
#ifdef DEFER_MISS
				if (f.template findBid<false>(c, x, y, cost, beta)) auctionBidding<false>(x, y.first, cost, u_count, unassigned, r_count, receiving, B, Block);
				else
				{
#pragma omp critical
					enqueue(x, d_count, deferred);
				}
#else
				f.template findBid<false>(c, x, y, cost, beta);
				auctionBidding<false>(x, y, cost, u_count, unassigned, r_count, receiving, B, Block);
#endif
			}
		}
		else
		{
			for (int xi = 0; xi < uc; xi++)
			{
				int x = unassigned[1][xi];
				std::pair<int, int> y;
				std::pair<AC, AC> cost;
#ifdef DEFER_MISS
				if (f.template findBid<true>(c, x, y, cost, beta)) auctionBidding<true>(x, y.first, cost, u_count, unassigned, r_count, receiving, B, Block);
				else enqueue(x, d_count, deferred);
#else
				f.template findBid<true>(c, x, y, cost, beta);
				auctionBidding<true>(x, y, cost, u_count, unassigned, r_count, receiving, B, Block);
#endif
			}
		}

#ifdef DEFER_MISS
		const int dc = d_count;

#ifdef DISPLAY_MISS
		hit_count += uc - dc;
		miss_count += dc;
#endif
		if (dc > 0)
		{
			if ((5 * dc >= 4 * processor_count) || (dc >= processor_count * 12))
			{
#pragma omp parallel
				{
#pragma omp for schedule(dynamic)
					for (int xi = 0; xi < dc; xi++)
					{
						int x = deferred[xi];
						std::pair<int, int> y(-1, -1);
						std::pair<AC, AC> cost;
						f.template fillCache<false, false>(c, x, y, cost, beta);
						auctionBidding<false>(x, y.first, cost, u_count, unassigned, r_count, receiving, B, Block);
					}
				}
			}
			else
			{
				for (int xi = 0; xi < dc; xi++)
				{
					int x = deferred[xi];
					std::pair<int, int> y(-1, -1);
					std::pair<AC, AC> cost;
					f.template fillCache<true, false>(c, x, y, cost, beta);
					auctionBidding<true>(x, y.first, cost, u_count, unassigned, r_count, receiving, B, Block);
				}
			}
		}
#endif

		int modified = r_count;
		auctionBuying<EXACT>(coupling, beta, u_count, unassigned, r_count, receiving, B, epsilon);
		// always required
		c.updateBeta(beta, modified);

		return target_size - u_count[0];
	}

	template <class COST, class FIND>
	int assign(
		std::vector<int> &coupling, std::vector<AC> &beta,
		COST &c, FIND &f,
		std::vector<int> &u_count, std::vector<std::vector<int>> &unassigned,
#ifdef DEFER_MISS
		int &d_count, std::vector<int> &deferred,
#endif
		int target_size, bool update)
	{
		std::swap(unassigned[0], unassigned[1]);
		u_count[1] = u_count[0];
		u_count[0] = 0;
		const int uc = u_count[1];

#ifdef DEFER_MISS
		d_count = 0;
#endif

#ifdef DEFER_MISS
#pragma omp parallel for
#else
#pragma omp parallel for schedule(dynamic)
#endif
		for (int xi = 0; xi < uc; xi++)
		{
			int x = unassigned[1][xi];
			std::pair<int, int> y;
			std::pair<AC, AC> cost;
#ifdef DEFER_MISS
			if (f.template findBid<false>(c, x, y, cost, beta)) coupling[x] = y.first;
			else
			{
#pragma omp critical
				enqueue(x, d_count, deferred);
			}
#else
			f.template findBid<false>(c, x, y, cost, beta);
			auctionBidding<false>(x, y, cost, u_count, unassigned, r_count, receiving, B, Block);
#endif
		}

#ifdef DEFER_MISS
		const int dc = d_count;

#ifdef DISPLAY_MISS
		hit_count += uc - dc;
		miss_count += dc;
#endif
		if (dc > 0)
		{
			if ((5 * dc >= 4 * processor_count) || (dc >= processor_count * 12))
			{
#pragma omp parallel
				{
#pragma omp for schedule(dynamic)
					for (int xi = 0; xi < dc; xi++)
					{
						int x = deferred[xi];
						std::pair<int, int> y(-1, -1);
						std::pair<AC, AC> cost;
						f.template fillCache<false, false>(c, x, y, cost, beta);
						coupling[x] = y.first;
					}
				}
			}
			else
			{
				for (int xi = 0; xi < dc; xi++)
				{
					int x = deferred[xi];
					std::pair<int, int> y(-1, -1);
					std::pair<AC, AC> cost;
					f.template fillCache<true, false>(c, x, y, cost, beta);
					coupling[x] = y.first;
				}
			}
		}
#endif

		// always required
		c.updateBeta(beta, target_size);

		return target_size;
	}

#if 1
	template <class COST, class FIND>
	int auctionMultiple(
		std::vector<int> &coupling, std::vector<AC> &beta, 
		COST &c, FIND &f, 
		std::vector<int> &u_count, std::vector<std::vector<int>> &unassigned, std::vector<std::vector<int>> &unassigned_count,
		int &r_count, std::vector<int> &receiving,
		std::vector<std::pair<volatile int, volatile AC>> &B, std::vector<std::mutex> &Block, AC epsilon)
#else
	template <class COST, class FIND>
	int auctionMultiple(
		std::vector<int> &coupling, std::vector<AC> &beta,
		COST &c, FIND &f,
		int &u_count, std::vector<int> &unassigned, std::vector<int> &unassigned_count,
		int &r_count, std::vector<int> &receiving,
		std::vector<std::pair<volatile int, volatile AC>> &B, std::vector<std::mutex> &Block, AC epsilon)
#endif
	{
		m_real_pending = 0;
#if 1
		std::swap(unassigned[0], unassigned[1]);
		std::swap(unassigned_count[0], unassigned_count[1]);
		u_count[1] = u_count[0];
		u_count[0] = 0;
		const int uc = u_count[1];
#else
		const int uc = u_count;
#endif

		for (int xi = 0; xi < uc; xi++)
		{
			int x = unassigned[1][xi];
			std::pair<AC, AC> cost;
			f.template findBid<false>(c, x, m_y[0], cost, beta, coupling);
			auctionBiddingMultiple<false>(coupling, x, m_y[0], cost, u_count, unassigned, unassigned_count, r_count, receiving, B, Block);
			auctionBuyingMultiple(coupling, beta, u_count, unassigned, unassigned_count, r_count, receiving, B, epsilon);
		}

		if (((u_count[0] == 0) && (u_count[0] < m_real_pending)) || (u_count[0] > m_real_pending))
		{
			std::cout << "Error: " << u_count[0] << " vs. " << m_real_pending << std::endl;
			exit(-1);
		}

		return (int)(coupling.size() - m_real_pending);
	}
};
