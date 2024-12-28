// Copyright 2024 Ivanov Mike
#pragma once

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/vector.hpp>
#include <cfloat>
#include <cmath>
#include <functional>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"

namespace ivanov_m_optimization_by_characteristics_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_,
                                 std::function<double(double, double)> f_,
                                 std::vector<std::function<bool(double, double)>> restriction_)
      : Task(std::move(taskData_)), f(std::move(f_)), restriction(std::move(restriction_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double centerX, centerY;                                       // coordinates of the search area center
  int size;                                                      // size in points
  double step;                                                   // length of step
  double approximation;                                          // approximation of the result
  std::function<double(double, double)> f;                       // function for optimization
  std::vector<std::function<bool(double, double)>> restriction;  // storage of function which restrict X and Y
  double res;                                                    // storage of result: X, Y, value
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::function<double(double, double)> f_,
                               std::vector<std::function<bool(double, double)>> restriction_)
      : Task(std::move(taskData_)), f(std::move(f_)), restriction(std::move(restriction_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double centerX, centerY;                                       // coordinates of the search area center
  int size;                                                      // size in points
  double step;                                                   // length of step
  double approximation;                                          // approximation of the result
  std::function<double(double, double)> f;                       // function for optimization
  std::vector<std::function<bool(double, double)>> restriction;  // storage of function which restrict X and Y
  double res;                                                    // storage of result: X, Y, value
  boost::mpi::communicator world;
};

}  // namespace ivanov_m_optimization_by_characteristics_mpi