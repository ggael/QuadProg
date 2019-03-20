// This file is part of QuadProgPP
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef QUADPROGPP_H
#define QUADPROGPP_H

#include <Eigen/Core>

namespace QuadProgPP{
  using namespace Eigen;

  struct Scheduling
  {
    enum ModeType {
      WorstFirst,     // Default strategy: check all inequalities, pick the worst one (if any), solve it
      WorstSetFirst,  // chek all inequalities, and pick about "initial_chunk_size" worst ones, and then satisfy them all using the default strategy
      SlidingWindows, // check the current chunk of initial_chunk_size inequalities, pick and solve the worst one (if any), shift the next chunk and repeat until all chunks are satisfied
    };

    ModeType mode;
    int initial_chunk_size;
    double grow_factor;

    Scheduling()
      : mode(WorstFirst), initial_chunk_size(-1), grow_factor(1)
    {}

    void initSlidingWindows(int n, int m)
    {
      mode = SlidingWindows;
      initial_chunk_size = std::min(std::max(n/2,m/64+1),m);
      grow_factor = 1;
    }

    void initWorstSetFirst(int n, int m)
    {
      mode = WorstSetFirst;
      initial_chunk_size = std::min(2*n,m);
      grow_factor = 1;
    }

    static void shift_window(int &chunk_start, int &chunk_size, int m)
    {
      if(chunk_start+chunk_size==m)
        chunk_start = 0;
      else
      {
        chunk_start += chunk_size;
        if(chunk_start+chunk_size>m)
          chunk_start = m-chunk_size;
      }
    }
  };

  void init_qp(Ref<MatrixXd> G);

  // Shortcut for init_qp(G) followed by a solve of the quadratic energy, and a call to solve_quadprog_with_guess().
  double solve_quadprog(Ref<MatrixXd> G, Ref<const VectorXd> g0,
                        Ref<const MatrixXd> CE, Ref<const VectorXd> ce0,
                        Ref<const MatrixXd> CI, Ref<const VectorXd> ci0,
                        Ref<VectorXd> x);

  /**
  * \param L Cholesky factor of the quadratic objective as computed by init_qp
  * \param g0 linear part of the quadratic objective
  * \param x on input: initial solution, on output the optimal solution (if any)
  * \param scheduling strategy used to prioritize inequalities
  * \param active_set if not null, treats the indexed inequality first, and on output active_set is filled with the remaining active-set
  *
  * Solves the following quadratic problem using Goldfarb and Idnani's algorithm  (dual active-set method) [1].
  *
  *   minimizes 1/2 x^T G x + x^T g0
  *   s.t.
  *   CE^T + ce0 =  0
  *   CI^T + ci0 >= 0
  *
  *  G is given as its Cholesky factor L (LL^T = G).
  *
  * [1] D. Goldfarb, A. Idnani. A numerically stable dual method for solving
  *     strictly convex quadratic programs. Mathematical Programming 27 (1983) pp. 1-33.
  */
  double solve_quadprog_with_guess(Ref<const MatrixXd> L, Ref<const VectorXd> g0,
                                   Ref<const MatrixXd> CE, Ref<const VectorXd> ce0,
                                   Ref<const MatrixXd> CI, Ref<const VectorXd> ci0,
                                   Ref<VectorXd> x,
                                   Scheduling scheduling = Scheduling(),
                                   VectorXi *active_set = 0);
}

#endif // #define QUADPROGPP_H
