// Copyright 2024 Ivanov Mike
#pragma once

#include <cmath>
#include <functional>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"

namespace ivanov_m_optimization_by_characteristics_seq {
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::function<double(double, double)> f_,
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

}  // namespace ivanov_m_optimization_by_characteristics_seq