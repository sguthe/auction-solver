#pragma once

#include "../lap_solver.h"

namespace lap
{
	namespace omp
	{
		template <class SC, class I>
		std::pair<SC, SC> estimateEpsilon(int dim, int dim2, I& iterator, SC *v, int *perm)
		{
#ifdef LAP_DEBUG
			auto start_time = std::chrono::high_resolution_clock::now();
#endif
			SC *mod_v;
			int *picked;
			SC *merge_cost;
			int *merge_idx;
			SC *v2;

			lapAlloc(mod_v, dim2, __FILE__, __LINE__);
			lapAlloc(v2, dim2, __FILE__, __LINE__);
			lapAlloc(picked, dim2, __FILE__, __LINE__);
			lapAlloc(merge_cost, omp_get_max_threads() << 3, __FILE__, __LINE__);
			lapAlloc(merge_idx, omp_get_max_threads() << 3, __FILE__, __LINE__);

			SC lower_bound = SC(0);
			SC upper_bound = SC(0);
			SC greedy_bound = SC(0);

			memset(picked, 0, sizeof(int) * dim2);

#pragma omp parallel
			{
				int t = omp_get_thread_num();
				int threads = omp_get_num_threads();
				for (int i = 0; i < dim2; i++)
				{
					SC min_cost_l, max_cost_l, picked_cost_l;
					int j_min;
					if (i < dim)
					{
						const auto *tt = iterator.getRow(t, i);
#pragma omp barrier
						auto cost = [&tt](int j) -> SC { return (SC)tt[j]; };
						getMinMaxBest(min_cost_l, max_cost_l, picked_cost_l, j_min, cost, picked + iterator.ws.part[t].first, iterator.ws.part[t].second - iterator.ws.part[t].first);
						j_min += iterator.ws.part[t].first;
						// a little hacky
						if ((i >= iterator.ws.part[t].first) && (i < iterator.ws.part[t].second))
						{
							merge_cost[1] = cost(i - iterator.ws.part[t].first);
						}
						merge_cost[(t << 3)] = min_cost_l;
						merge_cost[(t << 3) + 2] = picked_cost_l;
						merge_idx[(t << 3)] = j_min;
#pragma omp barrier
						min_cost_l = merge_cost[0];
						max_cost_l = merge_cost[1];
						picked_cost_l = merge_cost[2];
						j_min = merge_idx[0];
						for (int ii = 1; ii < threads; ii++)
						{
							min_cost_l = std::min(min_cost_l, merge_cost[(ii << 3)]);
							if (merge_cost[(ii << 3) + 2] < picked_cost_l)
							{
								picked_cost_l = merge_cost[(ii << 3) + 2];
								j_min = merge_idx[(ii << 3)];
							}
						}
						updateEstimatedV(v + iterator.ws.part[t].first, mod_v + iterator.ws.part[t].first, cost, (i == 0), (i == 1), min_cost_l, max_cost_l, iterator.ws.part[t].second - iterator.ws.part[t].first);
					}
					else
					{
						auto cost = [](int j) -> SC { return SC(0); };
						getMinMaxBest(min_cost_l, max_cost_l, picked_cost_l, j_min, cost, picked + iterator.ws.part[t].first, iterator.ws.part[t].second - iterator.ws.part[t].first);
						j_min += iterator.ws.part[t].first;
						// a little hacky
						if ((i >= iterator.ws.part[t].first) && (i < iterator.ws.part[t].second))
						{
							merge_cost[1] = cost(i - iterator.ws.part[t].first);
						}
						merge_cost[(t << 3)] = min_cost_l;
						merge_cost[(t << 3) + 2] = picked_cost_l;
						merge_idx[(t << 3)] = j_min;
#pragma omp barrier
						min_cost_l = merge_cost[0];
						max_cost_l = merge_cost[1];
						picked_cost_l = merge_cost[2];
						j_min = merge_idx[0];
						for (int ii = 1; ii < threads; ii++)
						{
							min_cost_l = std::min(min_cost_l, merge_cost[(ii << 3)]);
							if (merge_cost[(ii << 3) + 2] < picked_cost_l)
							{
								picked_cost_l = merge_cost[(ii << 3) + 2];
								j_min = merge_idx[(ii << 3)];
							}
						}
						updateEstimatedV(v + iterator.ws.part[t].first, mod_v + iterator.ws.part[t].first, cost, (i == 0), (i == 1), min_cost_l, max_cost_l, iterator.ws.part[t].second - iterator.ws.part[t].first);
					}
					if (t == 0)
					{
						picked[j_min] = 1;
						lower_bound += min_cost_l;
						upper_bound += max_cost_l;
						greedy_bound += picked_cost_l;
					}
				}
			}
			// make sure all j are < 0
			normalizeV(v, dim2);

			greedy_bound = std::min(greedy_bound, upper_bound);

			SC initial_gap = upper_bound - lower_bound;
			SC greedy_gap = greedy_bound - lower_bound;
			SC initial_greedy_gap = greedy_gap;

#ifdef LAP_DEBUG
			{
				std::stringstream ss;
				ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " initial_gap = " << initial_gap;
				lap::displayTime(start_time, ss.str().c_str(), lapDebug);
			}
			{
				std::stringstream ss;
				ss << "upper_bound = " << greedy_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap;
				lap::displayTime(start_time, ss.str().c_str(), lapDebug);
			}
#endif

			SC upper = std::numeric_limits<SC>::max();
			SC lower;

			memset(picked, 0, sizeof(int) * dim2);

			lower_bound = SC(0);
			upper_bound = SC(0);

#pragma omp parallel
			{
				int t = omp_get_thread_num();
				int threads = omp_get_num_threads();
				// reverse order
				for (int i = dim2 - 1; i >= 0; --i)
				{
					SC min_cost_l, second_cost_l, picked_cost_l;
					int j_min;
					if (i < dim)
					{
						const auto *tt = iterator.getRow(t, i);
						auto cost = [&tt, &v, &iterator, &t](int j) -> SC { return (SC)tt[j] - v[j + iterator.ws.part[t].first]; };
						getMinSecondBest(min_cost_l, second_cost_l, picked_cost_l, j_min, cost, picked + iterator.ws.part[t].first, iterator.ws.part[t].second - iterator.ws.part[t].first);
					}
					else
					{
						auto cost = [&v, &iterator, &t](int j) -> SC { return -v[j + iterator.ws.part[t].first]; };
						getMinSecondBest(min_cost_l, second_cost_l, picked_cost_l, j_min, cost, picked + iterator.ws.part[t].first, iterator.ws.part[t].second - iterator.ws.part[t].first);
					}
					j_min += iterator.ws.part[t].first;
#pragma omp barrier
					merge_cost[(t << 3)] = min_cost_l;
					merge_cost[(t << 3) + 1] = second_cost_l;
					merge_cost[(t << 3) + 2] = picked_cost_l;
					merge_idx[(t << 3)] = j_min;
#pragma omp barrier
					if (t == 0)
					{
						for (int ii = 1; ii < threads; ii++)
						{
							if (merge_cost[(ii << 3)] < min_cost_l)
							{
								second_cost_l = std::min(min_cost_l, merge_cost[(ii << 3) + 1]);
								min_cost_l = merge_cost[(ii << 3)];
							}
							else
							{
								second_cost_l = std::min(second_cost_l, merge_cost[(ii << 3)]);
							}
							if (merge_cost[(ii << 3) + 2] < picked_cost_l)
							{
								picked_cost_l = merge_cost[(ii << 3) + 2];
								j_min = merge_idx[(ii << 3)];
							}
						}
						perm[i] = i;
						picked[j_min] = 1;
						mod_v[i] = second_cost_l - min_cost_l;
						// need to use the same v values in total
						lower_bound += min_cost_l + v[j_min];
						upper_bound += picked_cost_l + v[j_min];
					}
				}
			}

			upper_bound = greedy_bound = std::min(upper_bound, greedy_bound);

			greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
			{
				std::stringstream ss;
				ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap;
				lap::displayTime(start_time, ss.str().c_str(), lapDebug);
			}
#endif
			if (initial_gap < SC(4) * greedy_gap)
			{
				memcpy(v2, v, dim2 * sizeof(SC));
				// sort permutation by keys
				std::sort(perm, perm + dim, [&mod_v](int a, int b) { return (mod_v[a] > mod_v[b]) || ((mod_v[a] == mod_v[b]) && (a > b)); });

				lower_bound = SC(0);
				upper_bound = SC(0);
				// greedy search
				std::fill(mod_v, mod_v + dim2, SC(-1));
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					int threads = omp_get_num_threads();
					for (int i = 0; i < dim2; i++)
					{
						// greedy order
						int j_min;
						SC min_cost, min_cost_real;
						if (perm[i] < dim)
						{
							const auto *tt = iterator.getRow(t, perm[i]);
							auto cost = [&tt, &v, &iterator, &t](int j) -> SC { return (SC)tt[j] - v[j + iterator.ws.part[t].first]; };
							getMinimalCost(j_min, min_cost, min_cost_real, cost, mod_v + iterator.ws.part[t].first, iterator.ws.part[t].second - iterator.ws.part[t].first);
						}
						else
						{
							auto cost = [&v, &iterator, &t](int j) -> SC { return -v[j + iterator.ws.part[t].first]; };
							getMinimalCost(j_min, min_cost, min_cost_real, cost, mod_v + iterator.ws.part[t].first, iterator.ws.part[t].second - iterator.ws.part[t].first);
						}
						merge_cost[(t << 3)] = min_cost;
						merge_cost[(t << 3) + 1] = min_cost_real;
						merge_idx[(t << 3)] = j_min + iterator.ws.part[t].first;
#pragma omp barrier
						if (t == 0)
						{
							min_cost = merge_cost[0];
							min_cost_real = merge_cost[1];
							j_min = merge_idx[0];
							for (int ii = 1; ii < threads; ii++)
							{
								if (merge_cost[(ii << 3)] < min_cost)
								{
									min_cost = merge_cost[(ii << 3)];
									j_min = merge_idx[(ii << 3)];
								}
								min_cost_real = std::min(min_cost_real, merge_cost[(ii << 3) + 1]);
							}
							mod_v[j_min] = SC(0);
							upper_bound += min_cost + v[j_min];
							// need to use the same v values in total
							lower_bound += min_cost_real + v[j_min];
							picked[i] = j_min;
						}
#pragma omp barrier
					}
				}

				greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
				{
					std::stringstream ss;
					ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap;
					lap::displayTime(start_time, ss.str().c_str(), lapDebug);
				}
#endif

#pragma omp parallel
				{
					int t = omp_get_thread_num();
					// update v in reverse order
					for (int i = dim2 - 1; i >= 0; --i)
					{
#pragma omp barrier
						if (perm[i] < dim)
						{
							const auto *tt = iterator.getRow(t, perm[i]);
							if ((picked[i] >= iterator.ws.part[t].first) && (picked[i] < iterator.ws.part[t].second))
							{
								merge_cost[0] = (SC)tt[picked[i] - iterator.ws.part[t].first] - v[picked[i]];
								mod_v[picked[i]] = SC(-1);
							}
#pragma omp barrier
							SC min_cost = merge_cost[0];
							for (int j = iterator.ws.part[t].first; j < iterator.ws.part[t].second; j++)
							{
								if (mod_v[j] >= SC(0))
								{
									SC cost_l = (SC)tt[j - iterator.ws.part[t].first] - v[j];
									if (cost_l < min_cost) v[j] -= min_cost - cost_l;
								}
							}
						}
						else
						{
							if ((picked[i] >= iterator.ws.part[t].first) && (picked[i] < iterator.ws.part[t].second))
							{
								merge_cost[0] = -v[picked[i]];
								mod_v[picked[i]] = SC(-1);
							}
#pragma omp barrier
							SC min_cost = merge_cost[0];
							for (int j = iterator.ws.part[t].first; j < iterator.ws.part[t].second; j++)
							{
								if (mod_v[j] >= SC(0))
								{
									SC cost_l = -v[j];
									if (cost_l < min_cost) v[j] -= min_cost - cost_l;
								}
							}
						}
					}
				}

				normalizeV(v, dim2);

				SC old_upper_bound = upper_bound;
				SC old_lower_bound = lower_bound;
				upper_bound = SC(0);
				lower_bound = SC(0);
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					int threads = omp_get_num_threads();
					for (int i = 0; i < dim2; i++)
					{
						SC min_cost_real;
						if (perm[i] < dim)
						{
							const auto *tt = iterator.getRow(t, perm[i]);
							if ((picked[i] >= iterator.ws.part[t].first) && (picked[i] < iterator.ws.part[t].second))
							{
								upper_bound += (SC)tt[picked[i] - iterator.ws.part[t].first];
							}
							min_cost_real = std::numeric_limits<SC>::max();

							for (int j = iterator.ws.part[t].first; j < iterator.ws.part[t].second; j++)
							{
								SC cost_l = (SC)tt[j - iterator.ws.part[t].first] - v[j];
								min_cost_real = std::min(min_cost_real, cost_l);
							}
						}
						else
						{
							min_cost_real = std::numeric_limits<SC>::max();
							for (int j = iterator.ws.part[t].first; j < iterator.ws.part[t].second; j++) min_cost_real = std::min(min_cost_real, -v[j]);
						}
#pragma omp barrier
						merge_cost[t << 3] = min_cost_real;
#pragma omp barrier
						// bounds are relative to v
						if (t == 0)
						{
							for (int ii = 1; ii < threads; ii++) min_cost_real = std::min(min_cost_real, merge_cost[ii << 3]);
							lower_bound += min_cost_real + v[picked[i]];
						}
					}
				}
				upper_bound = std::min(upper_bound, old_upper_bound);
				lower_bound = std::max(lower_bound, old_lower_bound);
				greedy_gap = upper_bound - lower_bound;
				double ratio2 = (double)greedy_gap / (double)initial_greedy_gap;

#ifdef LAP_DEBUG
				{
					std::stringstream ss;
					ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << (double)greedy_gap / (double)initial_gap;
					lap::displayTime(start_time, ss.str().c_str(), lapDebug);
				}
#endif
				if (ratio2 > 1.0e-09)
				{
					for (int i = 0; i < dim2; i++)
					{
						v[i] = (SC)((double)v2[i] * ratio2 + (double)v[i] * (1.0 - ratio2));
					}
				}
			}

			getUpperLower(upper, lower, greedy_gap, initial_gap, dim, dim2);

			lapFree(mod_v);
			lapFree(picked);
			lapFree(merge_cost);
			lapFree(merge_idx);
			lapFree(v2);

			return std::pair<SC, SC>((SC)upper, (SC)lower);
		}

		__forceinline void dijkstraCheck(int& endofpath, bool& unassignedfound, int jmin, int** colsol, char** colactive, std::pair<int, int>* part)
		{
			int t = 0;
			while (jmin >= part[t].second) t++;
			int jmin_local = jmin - part[t].first;
			colactive[t][jmin_local] = 0;
			if (colsol[t][jmin_local] < 0)
			{
				endofpath = jmin;
				unassignedfound = true;
			}
		}

		__forceinline void resetRowColumnAssignment(int& endofpath, int f, int** pred, int* rowsol, int** colsol, std::pair<int, int>* part)
		{
			int i;
			do
			{
				int t = 0;
				while (endofpath >= part[t].second) t++;
				int endofpath_local = endofpath - part[t].first;
				i = pred[t][endofpath_local];
				colsol[t][endofpath_local] = i;
				std::swap(endofpath, rowsol[i]);
			} while (i != f);
		}

		template <class SC, class CF, class I>
		void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)

			// input:
			// dim        - problem size
			// costfunc - cost matrix
			// findcost   - searching cost matrix

			// output:
			// rowsol     - column assigned to row in solution
			// colsol     - row assigned to column in solution
			// u          - dual variables, row reduction numbers
			// v          - dual variables, column reduction numbers

		{
#ifndef LAP_QUIET
			auto start_time = std::chrono::high_resolution_clock::now();

			long long total_hit = 0LL;
			long long total_miss = 0LL;

			long long total_rows = 0LL;
			long long total_virtual = 0LL;

			long long last_rows = 0LL;
			long long last_virtual = 0LL;

			int elapsed = -1;
#else
#ifdef LAP_DISPLAY_EVALUATED
			long long total_hit = 0LL;
			long long total_miss = 0LL;

			long long total_rows = 0LL;
			long long total_virtual = 0LL;
#endif
#endif

			int  **pred;
			int  endofpath;
			char **colactive;
			SC **d;
			int **colsol;
			SC epsilon_upper;
			SC epsilon_lower;
			SC **v;
			int *perm;

#ifdef LAP_DEBUG
			std::vector<SC *> v_list;
			std::vector<SC> eps_list;
#endif

//			lapAlloc(colactive, dim2, __FILE__, __LINE__);
//			lapAlloc(d, dim2, __FILE__, __LINE__);
//			lapAlloc(pred, dim2, __FILE__, __LINE__);
//			lapAlloc(colsol, dim2, __FILE__, __LINE__);
//			lapAlloc(v, dim2, __FILE__, __LINE__);
			lapAlloc(colactive, omp_get_max_threads(), __FILE__, __LINE__);
			lapAlloc(d, omp_get_max_threads(), __FILE__, __LINE__);
			lapAlloc(pred, omp_get_max_threads(), __FILE__, __LINE__);
			lapAlloc(colsol, omp_get_max_threads(), __FILE__, __LINE__);
			lapAlloc(v, omp_get_max_threads(), __FILE__, __LINE__);
			lapAlloc(perm, dim2, __FILE__, __LINE__);

#pragma omp parallel
			{
				int t = omp_get_thread_num();
				int start = iterator.ws.part[t].first;
				int end = iterator.ws.part[t].second;
				int count = end - start;

				lapAlloc(colactive[t], count, __FILE__, __LINE__);
				lapAlloc(d[t], count, __FILE__, __LINE__);
				lapAlloc(pred[t], count, __FILE__, __LINE__);
				lapAlloc(colsol[t], count, __FILE__, __LINE__);
				lapAlloc(v[t], count, __FILE__, __LINE__);
			}

			SC *min_private;
			int *jmin_private;
			// use << 3 to avoid false sharing
			lapAlloc(min_private, omp_get_max_threads() << 3, __FILE__, __LINE__);
			lapAlloc(jmin_private, omp_get_max_threads() << 3, __FILE__, __LINE__);

#ifdef LAP_ROWS_SCANNED
			unsigned long long *scancount;
			unsigned long long *pathlength;
			lapAlloc(scancount, dim2, __FILE__, __LINE__);
			lapAlloc(pathlength, dim2, __FILE__, __LINE__);
			memset(scancount, 0, dim2 * sizeof(unsigned long long));
			memset(pathlength, 0, dim2 * sizeof(unsigned long long));
#endif

			SC epsilon;

			if (use_epsilon)
			{
				SC* tmp_v;
				lapAlloc(tmp_v, dim2, __FILE__, __LINE__);
				std::pair<SC, SC> eps = lap::omp::estimateEpsilon(dim, dim2, iterator, tmp_v, perm);
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;
					int count = end - start;
					memcpy(v[t], &(tmp_v[start]), count * sizeof(SC));
				}
				lapFree(tmp_v);
				epsilon_upper = eps.first;
				epsilon_lower = eps.second;
			}
			else
			{
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;
					int count = end - start;
					memset(v[t], 0, count * sizeof(SC));
				}
				epsilon_upper = SC(0);
				epsilon_lower = SC(0);
			}
			epsilon = epsilon_upper;

			bool first = true;
			bool second = false;
			bool reverse = true;

			if ((!use_epsilon) || (epsilon > SC(0)))
			{
				for (int i = 0; i < dim2; i++) perm[i] = i;
				reverse = false;
			}

			SC total_d = SC(0);
			SC total_eps = SC(0);
			while (epsilon >= SC(0))
			{
#ifdef LAP_DEBUG
				if (first)
				{
					SC *vv;
					lapAlloc(vv, dim2, __FILE__, __LINE__);
					v_list.push_back(vv);
					eps_list.push_back(epsilon);
					memcpy(v_list.back(), v, sizeof(SC) * dim2);
				}
#endif
				lap::getNextEpsilon(epsilon, epsilon_lower, total_d, total_eps, first, second, dim2);
				total_d = SC(0);
				total_eps = SC(0);
#ifndef LAP_QUIET
				{
					std::stringstream ss;
					ss << "eps = " << epsilon;
					const std::string tmp = ss.str();
					displayTime(start_time, tmp.c_str(), lapInfo);
				}
#endif
				// this is to ensure termination of the while statement
				if (epsilon == SC(0)) epsilon = SC(-1.0);
				memset(rowsol, -1, dim2 * sizeof(int));
				//memset(colsol, -1, dim2 * sizeof(int));

				int jmin;
				SC min, min_n;
				bool unassignedfound;

#ifndef LAP_QUIET
				int old_complete = 0;
#endif

#ifdef LAP_MINIMIZE_V
				//int dim_limit = ((reverse) || (epsilon < SC(0))) ? dim2 : dim;
				int dim_limit = dim2;
#else
				int dim_limit = dim2;
#endif

				// AUGMENT SOLUTION for each free row.
#ifndef LAP_QUIET
				displayProgress(start_time, elapsed, 0, dim2, " rows");
#endif
				jmin = dim2;
				min = std::numeric_limits<SC>::max();
				SC tt_jmin_global;

#pragma omp parallel
				{
					int t = omp_get_thread_num();
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;
					int count = end - start;

					char* colactive_private = colactive[t];
					SC* d_private = d[t];
					int* pred_private = pred[t];
					SC* v_private = v[t];
					int* colsol_private = colsol[t];

					for (int fc = 0; fc < dim_limit; fc++)
					{
						int f = perm[((reverse) && (fc < dim)) ? (dim - 1 - fc) : fc];
						int jmin_local = dim2;
						SC min_local = std::numeric_limits<SC>::max();
						if (f < dim)
						{
							auto tt = iterator.getRow(t, f);
							for (int j = 0; j < count; j++)
							{
								colactive_private[j] = 1;
								pred_private[j] = f;
								SC h = d_private[j] = tt[j] - v_private[j];
								if (h < min_local)
								{
									// better
									jmin_local = j;
									min_local = h;
								}
								// same, do only update if old was used and new is free
								else if ((h == min_local) && (colsol_private[jmin_local] >= 0) && (colsol_private[j] < 0)) jmin_local = j;
							}
						}
						else
						{
							min_local = std::numeric_limits<SC>::max();
							for (int j = 0; j < count; j++)
							{
								colactive_private[j] = 1;
								pred_private[j] = f;
								SC h = d_private[j] = -v_private[j];
								if (h < min_local)
								{
									// ignore any columns assigned to virtual rows
									if (colsol_private[j] < dim)
									{
										// better
										jmin_local = j;
										min_local = h;
									}
								}
								// same, do only update if old was used and new is free
								else if ((h == min_local) && (colsol_private[jmin_local] >= 0) && (colsol_private[j] < 0)) jmin_local = j;
							}
						}
						min_private[t << 3] = min_local;
						jmin_private[t << 3] = jmin_local;
#pragma omp barrier
						if (t == 0)
						{
#ifndef LAP_QUIET
							if (f < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
							if (f < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
							scancount[f]++;
#endif
							min = min_private[0];
							jmin = jmin_private[0];
							bool taken = false;
							for (int tt = 1; tt < omp_get_num_threads(); tt++)
							{
								int start = iterator.ws.part[tt].first;
								if (min_private[tt << 3] < min)
								{
									// better than previous
									min = min_private[tt << 3];
									jmin = jmin_private[tt << 3] + start;
									taken = (colsol[tt][jmin_private[tt << 3]] >= 0);
								}
								else if ((min_private[tt << 3] == min) && (taken) && (colsol[tt][jmin_private[tt << 3]] < 0))
								{
									jmin = jmin_private[tt << 3] + start;
									taken = false;
								}
							}
							unassignedfound = false;
							dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive, iterator.ws.part);
						}
						// marked skipped columns that were cheaper
						if (f >= dim)
						{
							for (int j = 0; j < count; j++)
							{
								// ignore any columns assigned to virtual rows
								if ((colsol_private[j] >= dim) && (d_private[j] <= min))
								{
									colactive_private[j] = 0;
								}
							}
						}
#pragma omp barrier
						while (!unassignedfound)
						{
							// update 'distances' between freerow and all unscanned columns, via next scanned column.
							int t_jmin = 0;
							while (jmin >= iterator.ws.part[t_jmin].second) t_jmin++;
							int idx_jmin = jmin - iterator.ws.part[t_jmin].first;
							int i = colsol[t_jmin][idx_jmin];
							jmin_local = dim2;
							min_local = std::numeric_limits<SC>::max();
							if (i < dim)
							{
								SC tt_jmin;
								SC v_jmin = v[t_jmin][idx_jmin];
								auto tt = iterator.getRow(t, i);
								if (t_jmin == t)
								{
									tt_jmin_global = tt_jmin = (SC)tt[jmin - start];
								}
#pragma omp barrier
								if (t_jmin != t)
								{
									tt_jmin = tt_jmin_global;
								}
								for (int j = 0; j < count; j++)
								{
									if (colactive_private[j] != 0)
									{
										SC v2 = (tt[j] - tt_jmin) - (v_private[j] - v_jmin) + min;
										SC h = d_private[j];
										if (v2 < h)
										{
											pred_private[j] = i;
											d_private[j] = v2;
											h = v2;
										}
										if (h < min_local)
										{
											// better
											jmin_local = j;
											min_local = h;
										}
										// same, do only update if old was used and new is free
										else if ((h == min_local) && (colsol_private[jmin_local] >= 0) && (colsol_private[j] < 0)) jmin_local = j;
									}
								}
							}
							else
							{
								SC v_jmin = v[t_jmin][idx_jmin];
								for (int j = 0; j < count; j++)
								{
									if (colactive_private[j] != 0)
									{
										SC v2 = -(v_private[j] - v_jmin) + min;
										SC h = d_private[j];
										if (v2 < h)
										{
											pred_private[j] = i;
											d_private[j] = v2;
											h = v2;
										}
										if (h < min_local)
										{
											// ignore any columns assigned to virtual rows
											if (colsol_private[j] < dim)
											{
												// better
												jmin_local = j;
												min_local = h;
											}
										}
										// same, do only update if old was used and new is free
										else if ((h == min_local) && (colsol_private[jmin_local] >= 0) && (colsol_private[j] < 0)) jmin_local = j;
									}
								}
							}
							min_private[t << 3] = min_local;
							jmin_private[t << 3] = jmin_local;
#pragma omp barrier
							if (t == 0)
							{
#ifndef LAP_QUIET
								if (i < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
								if (i < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
								scancount[i]++;
#endif
								min_n = min_private[0];
								jmin = jmin_private[0];
								bool taken = false;
								for (int tt = 1; tt < omp_get_num_threads(); tt++)
								{
									int start = iterator.ws.part[tt].first;
									if (min_private[tt << 3] < min_n)
									{
										// better than previous
										min_n = min_private[tt << 3];
										jmin = jmin_private[tt << 3] + start;
										taken = (colsol[tt][jmin_private[tt << 3]] >= 0);
									}
									else if ((min_private[tt << 3] == min_n) && (taken) && (colsol[tt][jmin_private[tt << 3]] < 0))
									{
										jmin = jmin_private[tt << 3] + start;
										taken = false;
									}
								}
								min = std::max(min, min_n);
								dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive, iterator.ws.part);
							}
							// marked skipped columns that were cheaper
							if (i >= dim)
							{
								for (int j = 0; j < count; j++)
								{
									// ignore any columns assigned to virtual rows
									if ((colactive_private[j] == 1) && (colsol_private[j] >= dim) && (d_private[j] <= min))
									{
										colactive_private[j] = 0;
									}
								}
							}
#pragma omp barrier
						}
						// update column prices. can increase or decrease
						if (epsilon > SC(0))
						{
							min_private[t << 3] = SC(0);
							min_private[(t << 3) + 1] = SC(0);
							updateColumnPrices(colactive_private, 0, count, min, v_private, d_private, epsilon, min_private[t << 3], min_private[(t << 3) + 1]);
						}
						else
						{
							updateColumnPrices(colactive_private, 0, count, min, v_private, d_private);
						}
#pragma omp barrier
						if (t == 0)
						{
							if (epsilon > SC(0)) for (int tt = 0; tt < omp_get_num_threads(); tt++)
							{
								total_d += min_private[tt << 3];
								total_eps += min_private[(tt << 3) + 1];
							}
#ifdef LAP_ROWS_SCANNED
							{
								int i;
								int eop = endofpath;
								do
								{
									i = pred[eop];
									eop = rowsol[i];
									if (i != f) pathlength[f]++;
								} while (i != f);
							}
#endif

							// reset row and column assignments along the alternating path.
							resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol, iterator.ws.part);
							// for next iteration
							jmin = dim2;
							min = std::numeric_limits<SC>::max();
#ifndef LAP_QUIET
							int level;
							if ((level = displayProgress(start_time, elapsed, fc + 1, dim_limit, " rows")) != 0)
							{
								long long hit, miss;
								iterator.getHitMiss(hit, miss);
								total_hit += hit;
								total_miss += miss;
								if ((hit != 0) || (miss != 0))
								{
									{
										if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
										else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
									}
								}
								old_complete = f + 1;
							}
#endif
						}
#pragma omp barrier
					}
				}

#ifdef LAP_MINIMIZE_V
				if (epsilon > SC(0))
				{
#if 0
					if (dim_limit < dim2) normalizeV(v, dim2, colsol);
					else lap::normalizeV(v, dim2);
#else
					if (dim_limit < dim2)
					{
						for (int j = 0; j < omp_get_max_threads(); j++)
						{
							for (int i = 0; i < iterator.ws.part[j].second - iterator.ws.part[j].first; i++)
							{
								if (colsol[j][i] < 0) v[j][i] -= SC(2) * epsilon;
							}
						}
					}
					SC max_v = v[0][0];
					for (int j = 0; j < omp_get_max_threads(); j++)
					{
						for (int i = 0; i < iterator.ws.part[j].second - iterator.ws.part[j].first; i++)
						{
							max_v = std::max(max_v, v[j][i]);
						}
					}
					for (int j = 0; j < omp_get_max_threads(); j++)
					{
						for (int i = 0; i < iterator.ws.part[j].second - iterator.ws.part[j].first; i++)
						{
							v[j][i] = v[j][i] - max_v;
						}
					}
#endif
				}
#endif

#ifdef LAP_DEBUG
				if (epsilon > SC(0))
				{
					SC *vv;
					lapAlloc(vv, dim2, __FILE__, __LINE__);
					v_list.push_back(vv);
					eps_list.push_back(epsilon);
					memcpy(v_list.back(), v, sizeof(SC) * dim2);
				}
				else
				{
					int count = (int)v_list.size();
					if (count > 0)
					{
						for (int l = 0; l < count; l++)
						{
							SC dlt(0), dlt2(0);
							for (int i = 0; i < dim2; i++)
							{
								SC diff = v_list[l][i] - v[i];
								dlt += diff;
								dlt2 += diff * diff;
							}
							dlt /= SC(dim2);
							dlt2 /= SC(dim2);
							lapDebug << "iteration = " << l << " eps/mse = " << eps_list[l] << " " << dlt2 - dlt * dlt << " eps/rmse = " << eps_list[l] << " " << sqrt(dlt2 - dlt * dlt) << std::endl;
							lapFree(v_list[l]);
						}
					}
				}
#endif
				second = first;
				first = false;
				reverse = !reverse;

#ifndef LAP_QUIET
				lapInfo << "  rows evaluated: " << total_rows;
				if (last_rows > 0LL) lapInfo << " (+" << total_rows - last_rows << ")";
				last_rows = total_rows;
				if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
				if (last_virtual > 0LL) lapInfo << " (+" << total_virtual - last_virtual << ")";
				last_virtual = total_virtual;
				lapInfo << std::endl;
				if ((total_hit != 0) || (total_miss != 0)) lapInfo << "  hit: " << total_hit << " miss: " << total_miss << std::endl;
#endif
			}

#ifdef LAP_QUIET
#ifdef LAP_DISPLAY_EVALUATED
			iterator.getHitMiss(total_hit, total_miss);
			lapInfo << "  rows evaluated: " << total_rows;
			if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
			lapInfo << std::endl;
			if ((total_hit != 0) || (total_miss != 0)) lapInfo << "  hit: " << total_hit << " miss: " << total_miss << std::endl;
#endif
#endif

#ifdef LAP_ROWS_SCANNED
			lapInfo << "row\tscanned\tlength" << std::endl;
			for (int f = 0; f < dim2; f++)
			{
				lapInfo << f << "\t" << scancount[f] << "\t" << pathlength[f] << std::endl;
			}

			lapFree(scancount);
			lapFree(pathlength);
#endif

			// free reserved memory.
			for (int t = 0; t < omp_get_max_threads(); t++)
			{
				lapFree(colactive[t]);
				lapFree(d[t]);
				lapFree(pred[t]);
				lapFree(colsol[t]);
				lapFree(v[t]);
			}
			lapFree(pred);
			lapFree(colactive);
			lapFree(d);
			lapFree(v);
			lapFree(colsol);
			lapFree(min_private);
			lapFree(jmin_private);
			lapFree(perm);
		}

		// shortcut for square problems
		template <class SC, class CF, class I>
		void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)
		{
			lap::omp::solve<SC>(dim, dim, costfunc, iterator, rowsol, use_epsilon);
		}

		template <class SC, class CF>
		SC cost(int dim, int dim2, CF &costfunc, int *rowsol)
		{
			SC total = SC(0);
#pragma omp parallel for reduction(+:total)
			for (int i = 0; i < dim; i++) total += costfunc.getCost(i, rowsol[i]);
			return total;
		}

		template <class SC, class CF>
		SC cost(int dim, CF &costfunc, int *rowsol)
		{
			return lap::omp::cost<SC, CF>(dim, dim, costfunc, rowsol);
		}
	}
}
