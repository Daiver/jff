// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)
#include "ceres/ceres.h"
#include "glog/logging.h"

#include <vector>
#include <stdlib.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
// Data generated using the following octave code.
//   randn('seed', 23497);
//   m = 0.3;
//   c = 0.1;
//   x=[0:0.075:5];
//   y = exp(m * x + c);
//   noise = randn(size(x)) * 0.2;
//   y_observed = y + noise;
//   data = [x', y_observed'];


template<typename T>
T func(const T m, const T c, const double x)
{
    return exp(m * T(x) + c) + x * x * log(m) * log(m) * c + c*c;
}

struct ExponentialResidual {
  ExponentialResidual(double x, double y)
      : x_(x), y_(y) {}
  template <typename T> bool operator()(const T* const m,
                                        const T* const c,
                                        T* residual) const {
    //residual[0] = T(y_) - exp(m[0] * T(x_) + c[0]);
    residual[0] = T(y_) - func(m[0], c[0], x_);
    return true;
  }
 private:
  const double x_;
  const double y_;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  const int kNumObservations = 50000;
  std::vector<double> data;

  double tM = 3.34;
  double tC = -4.32;

  srand(42);
  data.resize(2*kNumObservations);
  for(int i = 0; i < kNumObservations; ++i){
    const double x = 5.0/kNumObservations * i;
    const double y = func(tM, tC, x) * 
        (0.9 + 0.2 * rand() / RAND_MAX);
    data[2*i]     = x;
    data[2*i + 1] = y;

  }

  double m = 1.0;
  double c = 6.0;
  Problem problem;
  for (int i = 0; i < kNumObservations; ++i) {
    problem.AddResidualBlock(
        new AutoDiffCostFunction<ExponentialResidual, ceres::DYNAMIC, 1, 1>(
            new ExponentialResidual(data[2 * i], data[2 * i + 1]), 1),
        NULL,
        &m, &c);
  }
  Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  options.dense_linear_algebra_library_type = ceres::LAPACK;
  options.num_linear_solver_threads = 4;
  options.num_threads = 4;
  options.use_inner_iterations = true;
  //options.trust_region_strategy_type = ceres::DOGLEG;
  //options.dogleg_type = ceres::SUBSPACE_DOGLEG;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "Final   m: " << m << " c: " << c << "\n";
  return 0;
}

