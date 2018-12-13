#pragma once

#include <vector>

typedef float AuctionCost;

AuctionCost getCurrentCost(int *linear_index, float *linear_target, int target_size, float *linear_source, int source_size, int total_channel);
void auctionAlgorithm(int *linear_index, float *linear_target, int target_size, float *linear_source, int source_size, int total_channel, std::vector<AuctionCost> &beta, bool relaxed);
void auctionAlgorithmCleanup();
