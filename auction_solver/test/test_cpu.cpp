#ifdef _OPENMP
#  define LAP_OPENMP
#endif
#define LAP_QUIET
//#define LAP_DISPLAY_EVALUATED
//#define LAP_DEBUG
//#define LAP_NO_MEM_DEBUG
//#define LAP_ROWS_SCANNED

//#define RANDOM_SEED 1234

#include "../../lap_solver/lap.h"
#include "../auction/auction_cpu.h"
#include "../auction/auction_cost_adapter.h"

#include <random>
#include <string>
#include <fstream>
#include "test_options.h"

template <class CF> void testRandom(long long min_tab, long long max_tab, int runs, bool omp, bool caching, bool epsilon, bool sanity, std::string name_C);
template <class CF> void testGeometric(long long min_tab, long long max_tab, int runs, bool omp, bool caching, bool epsilon, bool disjoint, std::string name_C);
template <class CF> void testRandomLowRank(long long min_tab, long long max_tab, long long min_rank, long long max_rank, int runs, bool omp, bool caching, bool epsilon, std::string name_C);
template <class CF> void testImages(std::vector<std::string> &images, long long max_tab, int runs, bool omp, bool caching, bool epsilon, std::string name_C);

int main(int argc, char* argv[])
{
	Options opt;
	int r = opt.parseOptions(argc, argv);
	if (r != 0) return r;

	if (opt.use_omp)
	{
		// omp "warmup"
		int *tmp = new int[1024];
#pragma omp parallel for
		for (int i = 0; i < 1024; i++)
		{
			tmp[i] = -1;
		}
		delete[] tmp;
	}

	if (opt.use_double)
	{
			if (opt.use_single)
			{
				if (opt.run_sanity) testRandom<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, false, true, std::string("double"));
				if (opt.run_random) testRandom<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, false, false, std::string("double"));
				if (opt.run_random_low_rank) testRandomLowRank<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, opt.use_caching, false, std::string("double"));
				if (opt.run_geometric) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, false, false, std::string("double"));
				if (opt.run_geometric_disjoint) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, false, true, std::string("double"));
				if (opt.images.size() > 1) testImages<double>(opt.images, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, false, std::string("double"));
			}
			if (opt.use_epsilon)
			{
				if (opt.run_sanity) testRandom<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, true, true, std::string("double"));
				if (opt.run_random) testRandom<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, true, false, std::string("double"));
				if (opt.run_random_low_rank) testRandomLowRank<double>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, opt.use_caching, true, std::string("double"));
				if (opt.run_geometric) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, true, false, std::string("double"));
				if (opt.run_geometric_disjoint) testGeometric<double>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, true, true, std::string("double"));
				if (opt.images.size() > 1) testImages<double>(opt.images, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, true, std::string("double"));
			}
		}
	if (opt.use_float)
	{
		if (opt.use_single)
		{
			if (opt.run_sanity) testRandom<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, false, true, std::string("float"));
			if (opt.run_random) testRandom<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, false, false, std::string("float"));
			if (opt.run_random_low_rank) testRandomLowRank<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, opt.use_caching, false, std::string("float"));
			if (opt.run_geometric) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, false, false, std::string("float"));
			if (opt.run_geometric_disjoint) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, false, true, std::string("float"));
			if (opt.images.size() > 1) testImages<float>(opt.images, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, false, std::string("float"));
		}
		if (opt.use_epsilon)
		{
			if (opt.run_sanity) testRandom<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, true, true, std::string("float"));
			if (opt.run_random) testRandom<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, true, false, std::string("float"));
			if (opt.run_random_low_rank) testRandomLowRank<float>(opt.lap_min_tab, opt.lap_max_tab, opt.lap_min_rank, opt.lap_max_rank, opt.runs, opt.use_omp, opt.use_caching, true, std::string("float"));
			if (opt.run_geometric) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, true, false, std::string("float"));
			if (opt.run_geometric_disjoint) testGeometric<float>(opt.lap_min_tab, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, true, true, std::string("float"));
			if (opt.images.size() > 1) testImages<float>(opt.images, opt.lap_max_tab, opt.runs, opt.use_omp, opt.use_caching, true, std::string("float"));
		}
	}

	return 0;
}

template <class TC, class I>
TC guessEpsilon(int x_size, int y_size, I& iterator, int step = 1)
{
	TC epsilon(0);
	for (int x = 0; x < x_size; x += step)
	{
		const TC *tt = iterator.getRow(x);
		TC min_cost, max_cost;
		min_cost = max_cost = tt[0];
		for (int y = 1; y < y_size; y++)
		{
			TC cost_l = tt[y];
			min_cost = std::min(min_cost, cost_l);
			max_cost = std::max(max_cost, cost_l);
		}
		epsilon += max_cost - min_cost;
	}
	return (epsilon / TC(10 * (x_size + step - 1) / step));
}


template <class C, class M, class TP>
void testMatrix(int N, M &costMatrix, bool omp, bool caching, bool epsilon, TP &start_time, bool sanity = false)
{
	C eps = C(0);
	int *rowsol = new int[N];
	lap::DirectIterator<C, C, M> iterator(N, N, costMatrix);
	AdaptorCost<C, lap::DirectIterator<C, C, M>> adaptor(iterator, N);

	if (epsilon) eps = guessEpsilon<C>(N, N, iterator);
	std::vector<int> coupling(N);
	std::vector<C> beta;
	int cache_size = (int)ceil(sqrt((double)N / 10.0));

	lap::displayTime(start_time, "setup complete", std::cout);

	C cost(0);
	if (caching)
	{
		FindCaching<AdaptorCost<C, lap::DirectIterator<C, C, M>>, C> f(N, adaptor, beta, cache_size);
		cost = auctionSingle<C>(coupling, adaptor, f, beta, eps, [&]()
		{
#pragma omp parallel for
			for (int x = 0; x < N; x++)
			{
				// slack...
				if (coupling[x] != -1) rowsol[coupling[x]] = x;
			}
			C cost = getCurrentCost<C>(rowsol, adaptor, N);
			return cost;
		}, omp);
	}
	else
	{
		FindLinear<AdaptorCost<C, lap::DirectIterator<C, C, M>>, C> f(N);
		cost = auctionSingle<C>(coupling, adaptor, f, beta, eps, [&]()
		{
#pragma omp parallel for
			for (int x = 0; x < N; x++)
			{
				// slack...
				if (coupling[x] != -1) rowsol[coupling[x]] = x;
			}
			C cost = getCurrentCost<C>(rowsol, adaptor, N);
			return cost;
		}, omp);
	}

	{
		std::stringstream ss;
		ss << "cost = " << cost;
		lap::displayTime(start_time, ss.str().c_str(), std::cout);
	}

	if (sanity)
	{
		bool passed = true;
		for (long long i = 0; (passed) && (i < N); i++)
		{
			passed &= (rowsol[i] == i);
		}
		std::stringstream ss;
		if (passed) ss << "test passed: ";
		else ss << "test failed: ";
		C real_cost(0);
		for (int i = 0; i < N; i++) real_cost += adaptor.getCost(i, i);
		ss << "ground truth cost = " << real_cost;
		lap::displayTime(start_time, ss.str().c_str(), std::cout);
	}

	delete[] rowsol;
}

template <class C>
void testRandom(long long min_tab, long long max_tab, int runs, bool omp, bool caching, bool epsilon, bool sanity, std::string name_C)
{
	// random costs (directly supply cost matrix)
	for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));

			if (sanity) std::cout << "Sanity";
			else std::cout << "Random";
			std::cout << "<" << name_C << "> " << N << "x" << N;
			if (omp) std::cout << " multithreaded";
			if (epsilon)
			{
				std::cout << " with epsilon scaling";
				if (caching) std::cout << " and caching";
			}
			else if (caching) std::cout << " width caching";
			std::cout << std::endl;

			auto start_time = std::chrono::high_resolution_clock::now();

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
#ifdef RANDOM_SEED
			std::mt19937_64 generator(RANDOM_SEED);
#else
			std::random_device rd;
			std::mt19937_64 generator(rd());
#endif

			C *tab = new C[NN];
			if (sanity)
			{
				C *vec = new C[N << 1];
				for (long long i = 0; i < N << 1; i++) vec[i] = distribution(generator);
				if (omp)
				{
#ifdef LAP_OPENMP
#pragma omp parallel for
					for (int i = 0; i < N; i++)
					{
						int j;
						long long ii = (long long)i * (long long)N;
						for (j = 0; j < i; j++) tab[ii + j] = vec[i] + vec[j + N] + C(0.1);
						tab[ii + i] = vec[i] + vec[i + N];
						for (j = i + 1; j < N; j++) tab[ii + j] = vec[i] + vec[j + N] + C(0.1);
					}
#endif
				}
				else
				{
					for (int i = 0; i < N; i++)
					{
						int j;
						long long ii = (long long)i * (long long)N;
						for (j = 0; j < i; j++) tab[ii + j] = vec[i] + vec[j + N] + C(0.1);
						tab[ii + i] = vec[i] + vec[i + N];
						for (j = i + 1; j < N; j++) tab[ii + j] = vec[i] + vec[j + N] + C(0.1);
					}
				}
				delete[] vec;
			}
			else
			{
				for (long long i = 0; i < NN; i++) tab[i] = distribution(generator);
			}

			lap::TableCost<C> costMatrix(N, N, tab);

			testMatrix<C>(N, costMatrix, omp, caching, epsilon, start_time, sanity);

			delete[] tab;
		}
	}
}

template <class C>
void testRandomLowRank(long long min_tab, long long max_tab, long long min_rank, long long max_rank, int runs, bool omp, bool caching, bool epsilon, std::string name_C)
{
	// random costs (directly supply cost matrix)
	for (long long rank = min_rank; rank <= max_rank; rank <<= 1)
	{
		for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
		{
			for (int r = 0; r < runs; r++)
			{
				int N = (int)floor(sqrt((double)NN));

				std::cout << "RandomLowRank<" << name_C << "> " << N << "x" << N << " rank = " << rank;
				if (omp) std::cout << " multithreaded";
				if (epsilon)
				{
					std::cout << " with epsilon scaling";
					if (caching) std::cout << " and caching";
				}
				else if (caching) std::cout << " width caching";
				std::cout << std::endl;

				auto start_time = std::chrono::high_resolution_clock::now();

				std::uniform_real_distribution<C> distribution(0.0, 1.0);
#ifdef RANDOM_SEED
				std::mt19937_64 generator(RANDOM_SEED);
#else
				std::random_device rd;
				std::mt19937_64 generator(rd());
#endif

				// The following matrix will have at most the seletcted rank.
				C *vec = new C[N * rank];
				for (long long i = 0; i < rank; i++)
				{
					for (long long j = 0; j < N; j++) vec[i * N + j] = distribution(generator);
				}

				// cost function
				auto get_cost = [&vec, &N, &rank](int x, int y) -> C
				{
					C sum(0);
					for (long long k = 0; k < rank; k++)
					{
						sum += vec[k * N + x] * vec[k * N + y];
					}
					return sum / C(rank);
				};

				lap::SimpleCostFunction<C, decltype(get_cost)> costFunction(get_cost);
				lap::TableCost<C> costMatrix(N, N, costFunction);
				delete[] vec;

				testMatrix<C>(N, costMatrix, omp, caching, epsilon, start_time);
			}
		}
	}
}

template <class C>
void testGeometric(long long min_tab, long long max_tab, int runs, bool omp, bool caching, bool epsilon, bool disjoint, std::string name_C)
{
	// geometric costs in R^2 (supply function for calculating cost matrix)
	for (long long NN = min_tab * min_tab; NN <= max_tab * max_tab; NN <<= 1)
	{
		for (int r = 0; r < runs; r++)
		{
			int N = (int)floor(sqrt((double)NN));

			std::cout << "Geometric";
			if (disjoint) std::cout << " Disjoint";
			std::cout << " R^2<" << name_C << "> " << N << "x" << N;
			if (omp) std::cout << " multithreaded";
			if (epsilon)
			{
				std::cout << " with epsilon scaling";
				if (caching) std::cout << " and caching";
			}
			else if (caching) std::cout << " width caching";
			std::cout << std::endl;

			auto start_time = std::chrono::high_resolution_clock::now();

			std::uniform_real_distribution<C> distribution(0.0, 1.0);
#ifdef RANDOM_SEED
			std::mt19937_64 generator(RANDOM_SEED);
#else
			std::random_device rd;
			std::mt19937_64 generator(rd());
#endif

			C *tab_s = new C[2 * N];
			C *tab_t = new C[2 * N];

			for (int i = 0; i < 2 * N; i++)
			{
				tab_s[i] = distribution(generator);
				tab_t[i] = distribution(generator);
			}

			if (disjoint)
			{
				for (int i = 0; i < 2 * N; i += 2)
				{
					if (i < N)
					{
						tab_t[i] += C(1);
					}
					else
					{
						tab_s[i] += C(1);
						tab_s[i + 1] += C(1);
						tab_t[i + 1] += C(1);
					}
				}
			}

			// cost function
			auto get_cost = [&tab_s, &tab_t](int x, int y) -> C
			{
				int xx = x + x;
				int yy = y + y;
				C d0 = tab_s[xx] - tab_t[yy];
				C d1 = tab_s[xx + 1] - tab_t[yy + 1];
				return d0 * d0 + d1 * d1;
			};

			lap::SimpleCostFunction<C, decltype(get_cost)> costFunction(get_cost);
			lap::TableCost<C> costMatrix(N, costFunction);

			testMatrix<C>(N, costMatrix, omp, caching, epsilon, start_time);
		}
	}
}

class PPMImage
{
public:
	int width, height, max_val;
	unsigned char *raw;
public:
	PPMImage(std::string &fname) : width(0), height(0), max_val(0), raw(0)
	{
		std::ifstream in(fname.c_str(), std::ios::in | std::ios::binary);
		if (in.is_open()) {
			std::string line;

			std::getline(in, line);
			if (line != "P6") {
				std::cout << "\"" << fname << "\" in not a ppm P6 file." << std::endl;
				exit(-1);
			}
			do std::getline(in, line); while (line[0] == '#');
			{
				std::stringstream sline(line);
				sline >> width;
				sline >> height;
			}
			std::getline(in, line);
			{
				std::stringstream sline(line);
				sline >> max_val;
			}

			raw = new unsigned char[width * height * 3];

			in.read((char *)raw, width * height * 3);
		}
		else
		{
			std::cout << "Can't open \"" << fname << "\"" << std::endl;
			exit(-1);
		}
		in.close();
	}
	~PPMImage() { delete[] raw; }
};

template <class C> void testImages(std::vector<std::string> &images, long long max_tab, int runs, bool omp, bool caching, bool epsilon, std::string name_C)
{
	std::cout << "Comparing images ";
	if (omp) std::cout << " multithreaded";
	if (epsilon) std::cout << " with epsilon scaling";
	std::cout << std::endl;
	for (unsigned int a = 0; a < images.size() - 1; a++)
	{
		for (unsigned int b = a + 1; b < images.size(); b++)
		{
			PPMImage img_a(images[a]);
			PPMImage img_b(images[b]);
			std::cout << "Comparing image \"" << images[a] << "\" (" << img_a.width << "x" << img_a.height << ") with image \"" << images[b] << "\" (" << img_b.width << "x" << img_b.height << ")." << std::endl;
			for (int r = 0; r < runs; r++)
			{
				auto start_time = std::chrono::high_resolution_clock::now();

				// make sure img[0] is at most as large as img[1]
				PPMImage *img[2];
				if (img_a.width * img_a.height < img_b.width * img_b.height)
				{
					img[0] = &img_a;
					img[1] = &img_b;
				}
				else
				{
					img[0] = &img_b;
					img[1] = &img_a;
				}
				int N1 = img[0]->width * img[0]->height;
				int N2 = img[1]->width * img[1]->height;

				// cost function
				auto get_cost = [&img](int x, int y) -> C
				{
					C r = C(img[0]->raw[3 * x]) / C(img[0]->max_val) - C(img[1]->raw[3 * y]) / C(img[1]->max_val);
					C g = C(img[0]->raw[3 * x + 1]) / C(img[0]->max_val) - C(img[1]->raw[3 * y + 1]) / C(img[1]->max_val);
					C b = C(img[0]->raw[3 * x + 2]) / C(img[0]->max_val) - C(img[1]->raw[3 * y + 2]) / C(img[1]->max_val);
					C u = C(x % img[0]->width) / C(img[0]->width - 1) - C(y % img[1]->width) / C(img[1]->width - 1);
					C v = C(x / img[0]->width) / C(img[0]->height - 1) - C(y / img[1]->width) / C(img[1]->height - 1);
					return r * r + g * g + b * b + u * u + v * v;
				};

				int *rowsol = new int[N2];

				long long entries = (max_tab * max_tab) / N2;

				C eps = C(0);
				int N = std::max(N1, N2);

				std::vector<int> coupling(N);
				std::vector<C> beta;
				int cache_size = (int)ceil(sqrt((double)entries / 10.0));

				lap::SimpleCostFunction<C, decltype(get_cost)> costFunction(get_cost);

				if (N1 <= entries)
				{
					std::cout << "using table with " << N1 << " rows." << std::endl;

					lap::TableCost<C> costMatrix(N, costFunction);
					lap::DirectIterator<C, C, decltype(costMatrix)> iterator(N, N, costMatrix);
					AdaptorCost<C, lap::DirectIterator<C, C, lap::TableCost<C>>> adaptor(iterator, N);

					if (epsilon) eps = guessEpsilon<C>(N, N, iterator);

					lap::displayTime(start_time, "setup complete", std::cout);
					C cost(0);
					if (caching)
					{
						FindCaching<AdaptorCost<C, lap::DirectIterator<C, C, lap::TableCost<C>>>, C> f(N, adaptor, beta, cache_size);
						cost = auctionSingle<C>(coupling, adaptor, f, beta, eps, [&]()
						{
#pragma omp parallel for
							for (int x = 0; x < N; x++)
							{
								// slack...
								if (coupling[x] != -1) rowsol[coupling[x]] = x;
							}
							C cost = getCurrentCost<C>(rowsol, adaptor, N);
							return cost;
						}, omp);
					}
					else
					{
						FindLinear<AdaptorCost<C, lap::DirectIterator<C, C, lap::TableCost<C>>>, C> f(N);
						cost = auctionSingle<C>(coupling, adaptor, f, beta, eps, [&]()
						{
#pragma omp parallel for
							for (int x = 0; x < N; x++)
							{
								// slack...
								if (coupling[x] != -1) rowsol[coupling[x]] = x;
							}
							C cost = getCurrentCost<C>(rowsol, adaptor, N);
							return cost;
						}, omp);
					}

					{
						std::stringstream ss;
						ss << "cost = " << cost;
						lap::displayTime(start_time, ss.str().c_str(), std::cout);
					}
				}
				else
				{
				}
			}
		}
	}
}
