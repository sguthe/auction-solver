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

namespace auction
{
  int src_idx(int x, int source_size, int target_size)
  {
    if (source_size == target_size) return x;
    return (int)(((long long)x * (long long)(source_size - 1) + (long long)((target_size - 1) >> 1)) / (long long)(target_size - 1));
  }
}
