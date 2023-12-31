---
abstract:
  This document contains the instructions for building and testing the
  software bundle as well as instructions on how to use the software in
  your own project.
  Note that this software is the optimized auction solver that is used
  for comparison in the paper only. While it performs very well for small
  and purely random problems, it does not scale well beyond a certain
  point.
author:
  Stefan Guthe, Daniel Thuerck
title: "Algorithm 1015: A Fast Scalable Solver for the Dense Linear (Sum)
  Assignment Problem"
doi: https://dl.acm.org/doi/abs/10.1145/3442348
---

# Linux

This section contains the instructions for building and running under
Linux. For Windows, see below.

## Requirements

The following software is required to build the source code that comes
with this publication:

-   CMake 3.28

-   GCC

-   OpenMP (optional)

## Build Instructions

To build the test package that was used to generate all the performance
measurements in the main publication, run the following commands inside
the `auction_solver` directory:

-   `mkdir build`

-   `cd build`

-   `cmake ../gcc`

-   `cmake --build .`

-   `ctest`

# Windows

## Requirements

The following software is required to build the source code that comes
with this publication:

-   Visual Studio (at least Community Edition) 2014, 2017 or 2019

## Build Instructions

To build the test package that was used to generate all the performance
measurements in the main publication, the following steps are required:

-   load solution from `vc14`, `vc17` or `vc19`

-   press `Ctrl+Shift+B`

-   as an alternative, use the cmake build as described above.

# Running Tests

The following commands will re-produce the data found in the figures and
tables of the paper (Figure 1, 2 & 3 require special debug builds) using
Linux. For Windows, the relative path is different
(`../../../../../images/`, when executing in the build directory).

- Table 2:\
    `./test_cpu -table_min 1000 -table_max 128000 -random -geometric -geometric_disjoint -sanity -double -epsilon -caching -omp`\
    `./test_cpu -table_min 1000 -table_max 64000 -random -geometric -geometric_disjoint -sanity -double -epsilon -omp`

# Own Project

In order to use the software package in your own project, you need to
include the `auction.h` file after setting the desired defines in your
project. To get the same behaviour as the `test_cpu` program, use the
following:

    // enable OpenMP support
    #ifdef _OPENMP
    #  define LAP_OPENMP
    #endif
    // quiet mode
    #define LAP_QUIET

## High-Level Interface

Since this code was initially used for testing purposes only, there
is currently no high-level interface implemented.

## Low-Level Interface

The low-level interface can be found in the `auction.h` include files. The
single threaded CPU code consists of the following functions for solving
the linear assignment and calculating the final costs:

    namespace lap
    {
      template <class SC, class CF, class I> void solve(
        int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon = true, bool find_caching = true);
      template <class SC, class CF, class I> void solve(
        int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, 
        bool use_epsilon = true, bool find_caching = true);
      template <class SC, class CF> SC cost(
        int dim, CF &costfunc, int *rowsol);
      template <class SC, class CF> SC cost(
        int dim, int dim2, CF &costfunc, int *rowsol);
    }

The multi threaded CPU code uses the following interface:

    namespace lap
    {
      namespace omp
      {
        template <class SC, class CF, class I> void solve(
          int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon = true, bool find_caching = true);
        template <class SC, class CF, class I> void solve(
          int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, 
          bool use_epsilon = true, bool find_caching = true);
        template <class SC, class CF> SC cost(
          int dim, CF &costfunc, int *rowsol);
        template <class SC, class CF> SC cost(
          int dim, int dim2, CF &costfunc, int *rowsol);
      }
    }

Please refer to the same file for additional helper classes that can be
used for the low-level interface.
