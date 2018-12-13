#pragma once

#include "auction_helper.h"
#include "auction_cost_multiple.h"

#include "../kernel/kernel.h"
#include "../kernel/kernel_cu.h"
#include <omp.h>

#define USE_CUDA

class CudaRefs
{
public:
	void *d_target;
	void *d_source;
	void *d_cost[2];
	void *d_idx[2];
	void *d_beta;
	void *d_alpha;
	size_t temp_storage_bytes;
	void *d_temp_storage;
	bool d_beta_valid;
	bool d_alpha_valid;
	bool device_active;
};

template <class AC>
class DirectCost
{
private:
	int target_size;
	int total_channel;
	float *own_target;
	float *own_source;

	std::vector<CudaRefs> cRef;

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

		cudaDeviceProp deviceProp;
		cudaGetDeviceCount(&device_count);

		checkCudaErrors(cudaGetDeviceCount(&device_count));

		cRef.resize(device_count);

		for (int current_device = 0; current_device < device_count; current_device++)
		{
			cudaGetDeviceProperties(&deviceProp, current_device);

			// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
			cRef[current_device].device_active = (deviceProp.computeMode != cudaComputeModeProhibited);
		}

		float *h_source = new float[target_size * total_channel];
#pragma omp parallel for
		for (int x = 0; x < target_size; x++)
		{
			int x0 = src_idx(x, source_size, target_size);
			for (int cc = 0; cc < total_channel; cc++)
			{
				h_source[cc * target_size + x] = linear_source[cc * source_size + x0];
			}
		}

#pragma omp parallel
		{
			int t = omp_get_thread_num();
			if (cudaEnabled(t))
			{
				cudaSetDevice(t);
				allocDeviceData<float>(cRef[t].d_target, target_size * total_channel, __FILE__, __LINE__);
				allocDeviceData<float>(cRef[t].d_source, target_size * total_channel, __FILE__, __LINE__);
				allocDeviceData<float>(cRef[t].d_cost[0], target_size, __FILE__, __LINE__);
				allocDeviceData<float>(cRef[t].d_cost[1], target_size, __FILE__, __LINE__);
				allocDeviceData<int>(cRef[t].d_idx[0], target_size, __FILE__, __LINE__);
				allocDeviceData<int>(cRef[t].d_idx[1], target_size, __FILE__, __LINE__);
				allocDeviceData<float>(cRef[t].d_beta, target_size, __FILE__, __LINE__);
				allocDeviceData<float>(cRef[t].d_alpha, target_size, __FILE__, __LINE__);

				uploadData<float>(cRef[t].d_target, linear_target, target_size * total_channel, false);
				uploadData<float>(cRef[t].d_source, h_source, target_size * total_channel, false);
				cudaMemset(cRef[t].d_beta, 0, target_size * sizeof(float));
				cRef[t].d_beta_valid = true;
				cudaMemset(cRef[t].d_alpha, 0, target_size * sizeof(float));
				cRef[t].d_alpha_valid = true;

				cRef[t].temp_storage_bytes = 0;
				cRef[t].d_temp_storage = NULL;
			}
		}

		delete[] h_source;
	}

	__forceinline__ float update_target(float *linear_target, int x)
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

	__forceinline__ void update_target_cuda(int t, float *linear_target)
	{
		cudaSetDevice(t);
		uploadData<float>(cRef[t].d_target, linear_target, target_size * total_channel, false);
	}

	~DirectCost()
	{
		delete[] own_target;
		delete[] own_source;

#pragma omp parallel
		{
			int t = omp_get_thread_num();
			if (cudaEnabled(t))
			{
				freeDeviceData(cRef[t].d_target);
				freeDeviceData(cRef[t].d_source);
				freeDeviceData(cRef[t].d_cost[0]);
				freeDeviceData(cRef[t].d_cost[1]);
				freeDeviceData(cRef[t].d_idx[0]);
				freeDeviceData(cRef[t].d_idx[1]);
				freeDeviceData(cRef[t].d_beta);
				freeDeviceData(cRef[t].d_alpha);
				if (cRef[t].d_temp_storage != NULL) freeDeviceData(cRef[t].d_temp_storage);
			}
		}
	}

	__forceinline__ const AC getCost(int x, int y) const
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
	__forceinline__ void iterate(const C &c, int x,std::vector<AC> &beta)
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
	__forceinline__ void iterateReverse(const C &c, int x, AC &limit, std::vector<AC> &alpha)
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
	__forceinline__ void iterateReverse(const C &c, int x, std::vector<AC> &alpha)
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
	__forceinline__ void updateBeta(std::vector<AC> &beta, int r_count) {
		p_beta = &beta;
		for (int i = 0; i < (int)cRef.size(); i++) cRef[i].d_beta_valid = false;
	}
	__forceinline__ void updateAlpha(std::vector<AC> &alpha, int r_count) {
		p_alpha = &alpha;
		for (int i = 0; i < (int)cRef.size(); i++) cRef[i].d_alpha_valid = false;
	}

	__forceinline__ void printCases(bool reset) {}

#ifdef USE_CUDA
	__forceinline__ bool cudaEnabled(int t) { return (t < device_count) && cRef[t].device_active; }
#else
	__forceinline__ bool cudaEnabled(int t) { return false; }
#endif
	__forceinline__ void cudaFill(std::vector<int> &idx, std::vector<AC> &heap, int x)
	{
		int t = omp_get_thread_num();
		if (!cRef[t].d_beta_valid)
		{
			//uploadData<float>(d_beta, p_beta->data(), target_size, false);
			checkCudaErrors(cudaMemcpyAsync(cRef[t].d_beta, p_beta->data(), target_size * sizeof(float), cudaMemcpyHostToDevice));
			cRef[t].d_beta_valid = true;
		}

		//int *idx_raw = idx.data();
		//float *heap_raw = heap.data();

		initLinear(cRef[t].d_idx[0], target_size);
		fillCosts(cRef[t].d_cost[0], cRef[t].d_source, cRef[t].d_target, cRef[t].d_beta, x, target_size, total_channel, 0);
		cub_stable_sort_by_keys<float, int>(cRef[t].d_cost, cRef[t].d_idx, target_size, 0, cRef[t].d_temp_storage, cRef[t].temp_storage_bytes);
		//downloadData<int>(idx_raw, d_idx[0], (int) idx.size(), false);
		//downloadData<float>(heap_raw, d_cost[0], (int) heap.size(), false);
		checkCudaErrors(cudaMemcpyAsync(idx.data(), cRef[t].d_idx[0], idx.size() * sizeof(int), cudaMemcpyDeviceToHost));
		// we don't need the heap data
		//checkCudaErrors(cudaMemcpyAsync(heap.data(), cRef[t].d_cost[0], idx.size() * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaDeviceSynchronize());
	}

	__forceinline__ void cudaFillRow(int t, AC * data, int x, int start, int count)
	{
		fillCosts(cRef[t].d_cost[0], cRef[t].d_target, cRef[t].d_source, x, target_size, total_channel, start, count, 0);
		checkCudaErrors(cudaMemcpy(&(data[start]), cRef[t].d_cost[0], count * sizeof(float), cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaMemcpyAsync(&(data[start]), cRef[t].d_cost[0], count * sizeof(float), cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaStreamSynchronize(0));
	}

	__forceinline__ void cudaFillReverse(std::vector<int> &idx, std::vector<AC> &heap, int x)
	{
		int t = omp_get_thread_num();
		if (!cRef[t].d_alpha_valid)
		{
			//uploadData<float>(d_alpha, p_alpha->data(), target_size, false);
			checkCudaErrors(cudaMemcpyAsync(cRef[t].d_alpha, p_alpha->data(), target_size * sizeof(float), cudaMemcpyHostToDevice));
			cRef[t].d_alpha_valid = true;
		}

		//int *idx_raw = idx.data();
		//float *heap_raw = heap.data();

		initLinear(cRef[t].d_idx[0], target_size);
		fillCosts(cRef[t].d_cost[0], cRef[t].d_target, cRef[t].d_source, cRef[t].d_alpha, x, target_size, total_channel, 0);
		cub_stable_sort_by_keys<float, int>(cRef[t].d_cost, cRef[t].d_idx, target_size, 0, cRef[t].d_temp_storage, cRef[t].temp_storage_bytes);
		//downloadData<int>(idx_raw, d_idx[0], (int) idx.size(), false);
		//downloadData<float>(heap_raw, d_cost[0], (int) heap.size(), false);
		checkCudaErrors(cudaMemcpyAsync(idx.data(), cRef[t].d_idx[0], idx.size() * sizeof(int), cudaMemcpyDeviceToHost));
		// we don't need the heap data
		//checkCudaErrors(cudaMemcpyAsync(heap.data(), cRef[t].d_cost[0], idx.size() * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaDeviceSynchronize());
	}

	__forceinline__ void cudaFillColumn(int t, AC * data, int x, int start, int count)
	{

		fillCosts(cRef[t].d_cost[0], cRef[t].d_source, cRef[t].d_target, x, target_size, total_channel, start, count, 0);
		checkCudaErrors(cudaMemcpy(&(data[start]), cRef[t].d_cost[0], count * sizeof(float), cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaMemcpyAsync(&(data[start]), cRef[t].d_cost[0], count * sizeof(float), cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaStreamSynchronize(0));
	}

	__forceinline__ bool requiresCaching() { return true; }
};

template <class AC>
class DirectCostMultiple
{
private:
	CostMultiple<AC> &target;
	CostMultiple<AC> &source;
public:
	DirectCostMultiple(CostMultiple<AC> &target, CostMultiple<AC> &source) : target(target), source(source) { }

	__forceinline__ AC getCostMulti(int x, int y) const { return source.dist(y, target, x); }
	__forceinline__ AC getCostMulti2(int x, int y) const { return source.dist(y, target, getTargetMap(x)); }
	__forceinline__ AC getCost(int x, int y) const { return source.dist(getSourceMap(y), target, getTargetMap(x)); }
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
		for (int yy = 0; yy < (int)source.size(); yy++)
		{
			AC raw_cost = getCostMulti(x, yy);
			for (int yc = 0; yc < source.getCount(yy); yc++, yb++)
			{
				if (-beta[yy] <= limit)
				{
					AC ccost = raw_cost - beta[yb];
					c(yb, ccost);
				}
			}
		}
	}

	template <bool PAR, class C>
	__forceinline__ void iterateMulti(const C &c, int x, std::vector<AC> &beta)
	{
		int yb = 0;
		for (int yy = 0; yy < source.size(); yy++)
		{
			AC raw_cost = getCostMulti(x, yy);
			for (int yc = 0; yc < source.getCount(yy); yc++, yb++)
			{
				AC ccost = raw_cost - beta[yb];
				c(yb, ccost);
			}
		}
	}

	template <bool PAR, class C>
	__forceinline__ void iterate(const C &c, int x, AC &limit, std::vector<AC> &beta)
	{
		iterateMulti<PAR>(c, getTargetMap(x), limit, beta);
	}
	template <bool PAR, class C>
	__forceinline__ void iterate(const C &c, int x, std::vector<AC> &beta)
	{
		iterateMulti<PAR>(c, getTargetMap(x), beta);
	}
	__forceinline__ void updateBeta(std::vector<AC> &beta, int r_count) {}
	__forceinline__ void printCases(bool reset) {}
	__forceinline__ bool cudaEnabled(int t) { return false; }
	__forceinline__ void cudaFill(std::vector<int> &idx, std::vector<AC> &heap, int x) {}
};
