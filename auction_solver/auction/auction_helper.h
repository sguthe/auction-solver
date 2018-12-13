#pragma once

#if defined(_MSC_VER)
// use sprintf_s
#else
#define sprintf_s sprintf
#include <string.h>
#endif

#include <chrono>

#define MAX_COST 1.0e36f

#define MIN_EPSILON 0.0001f

#define DEFER_MISS
//#define DISPLAY_ALWAYS
//#define DISPLAY_THREAD_FILL
//#define DISPLAY_MISS

#ifdef DISPLAY_MISS
extern long long hit_count;
extern long long miss_count;

extern long long total_hit_count;
extern long long total_miss_count;
#endif

#ifdef DISPLAY_THREAD_FILL
extern std::vector<long long> thread_fill_count;
#endif

int src_idx(int x, int source_size, int target_size)
{
	if (source_size == target_size) return x;
	return (int)(((long long)x * (long long)(source_size - 1) + (long long)((target_size - 1) >> 1)) / (long long)(target_size - 1));
}

template <class TP>
void displayTime(TP &start_time, const char *msg)
{
	auto end_time = std::chrono::high_resolution_clock::now();
	long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
//	int elapsed = (int)((ms + 10000ll) / 10000ll);
	char time[256];
	long long sec = ms / 1000;
	ms -= sec * 1000;
	long long min = sec / 60;
	sec -= min * 60;
	long long hrs = min / 60;
	min -= hrs * 60;
	sprintf_s(time, "%3d:%02d:%02d.%03d", (int)hrs, (int)min, (int)sec, (int)ms);
	std::cout << time << ": " << msg << std::endl;
}

template <class TP>
bool checkTime(TP &start_time, int &elapsed)
{
	auto end_time = std::chrono::high_resolution_clock::now();
	long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

	if (elapsed * 10000 < ms)
	{
		elapsed = (int)((ms + 10000ll) / 10000ll);
		return true;
	}

	return false;
}

template <class TP, class AC>
void displayProgressBase(TP &start_time, int &elapsed, int completed, int target_size, int &iteration, AC epsilon, bool display, char *msg = NULL)
{
	auto end_time = std::chrono::high_resolution_clock::now();
	long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

	if (elapsed * 10000 < ms)
	{
		display = true;
	}

#ifdef DISPLAY_ALWAYS
	display = true;
#endif

	if (display)
	{
		elapsed = (int)((ms + 10000ll) / 10000ll);
		char time[256];
		long long sec = ms / 1000;
		ms -= sec * 1000;
		long long min = sec / 60;
		sec -= min * 60;
		long long hrs = min / 60;
		min -= hrs * 60;
		sprintf_s(time, "%3d:%02d:%02d.%03d", (int)hrs, (int)min, (int)sec, (int)ms);
#ifdef DISPLAY_MISS
		total_hit_count += hit_count;
		total_miss_count += miss_count;
#endif
		if (completed == target_size)
		{
			std::cout << time << ": solving " << completed << "/" << target_size << " iteration = " << iteration << " epsilon = " << epsilon;
			if (msg != NULL) std::cout << msg;
			std::cout << std::endl;
#ifdef DISPLAY_MISS
			if (hit_count + miss_count > 0)
			{
				std::cout << "    hit_count  = " << hit_count << " (" << (100.0 * (double)hit_count) / ((double)(hit_count + miss_count)) << "%)" << std::endl;
				std::cout << "    miss_count = " << miss_count << std::endl;
			}
			if (total_hit_count + total_miss_count > 0)
			{
				logger << "    total_hit_count  = " << total_hit_count << " (" << (100.0 * (double)total_hit_count) / ((double)(total_hit_count + total_miss_count)) << "%)" << std::endl;
				logger << "    total_miss_count = " << total_miss_count << std::endl;
			}
			total_hit_count = total_miss_count = 0ll;
#endif
			std::cout.flush();
		}
		else
		{
			std::cout << time << ": solving " << completed << "/" << target_size << " iteration = " << iteration << " epsilon = " << epsilon;
			if (msg != NULL) std::cout << msg;
			std::cout << std::endl;
#ifdef DISPLAY_MISS
			if (iteration > 0)
			{
				if (hit_count + miss_count > 0)
				{
					std::cout << "    hit_count  = " << hit_count << " (" << (100.0 * (double)hit_count) / ((double)(hit_count + miss_count)) << "%)" << std::endl;
					std::cout << "    miss_count = " << miss_count << std::endl;
				}
			}
#endif
		}
#ifdef DISPLAY_THREAD_FILL
		long long total = 0;
		for (int i = 0; i < thread_fill_count.size(); i++) total += thread_fill_count[i];
		if (total > 0)
		{
			for (int i = 0; i < thread_fill_count.size(); i++)
			{
				if (i == 0) std::cout << "    thread_counter = ";
				else std::cout << ", ";
				std::cout << thread_fill_count[i];
				double p = 100.0 * (double)thread_fill_count[i] / (double)total;
				std::cout << " (" << p << "%)";
				thread_fill_count[i] = 0;
			}
			std::cout << std::endl;
		}
#endif
#ifdef DISPLAY_MISS
		hit_count = miss_count = 0ll;
#endif
	}
}

template <class TP, class AC>
void displayProgress(TP &start_time, int &elapsed, int completed, int target_size, int &iteration, AC epsilon, AC &old_epsilon, bool display = false)
{
	if (epsilon != old_epsilon)
	{
		display = true;
		iteration = 0;
		old_epsilon = epsilon;
	}
	displayProgressBase(start_time, elapsed, completed, target_size, iteration, epsilon, display);
}

template <class TP, class AC>
void displayProgressCost(TP &start_time, int &elapsed, int completed, int target_size, int &iteration, AC epsilon, AC cost)
{
	char msg[1024];
	sprintf_s(msg, " EMD = %f", cost);
	displayProgressBase(start_time, elapsed, completed, target_size, iteration, epsilon, true, msg);
}

template <class TP>
bool displayProgress(TP &start_time, int &elapsed, int completed, int target_size, const char *msg = NULL, int iteration = -1, bool display = false)
{
	auto end_time = std::chrono::high_resolution_clock::now();
	long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

	if ((elapsed * 10000 < ms) || (completed == target_size))
	{
		display = true;
	}

#ifdef DISPLAY_ALWAYS
	display = true;
#endif

	if (display)
	{
		elapsed = (int)((ms + 10000ll) / 10000ll);
		char time[256];
		long long sec = ms / 1000;
		ms -= sec * 1000;
		long long min = sec / 60;
		sec -= min * 60;
		long long hrs = min / 60;
		min -= hrs * 60;
		sprintf_s(time, "%3d:%02d:%02d.%03d", (int)hrs, (int)min, (int)sec, (int)ms);
		if (completed == target_size)
		{
			logger << time << ": solving " << completed << "/" << target_size;
			if (iteration >= 0) logger << " iteration = " << iteration;
			if (msg != NULL) logger << msg;
			logger << std::endl;
			logger.flush();
		}
		else
		{
			std::cout << time << ": solving " << completed << "/" << target_size;
			if (iteration >= 0) std::cout << " iteration = " << iteration;
			if (msg != NULL) std::cout << msg;
			std::cout << std::endl;
		}
	}
	return display;
}
