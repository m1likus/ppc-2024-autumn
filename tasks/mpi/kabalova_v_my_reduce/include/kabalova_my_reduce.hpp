// Copyright 2024 Kabalova Valeria
#pragma once

#include <gtest/gtest.h>
//#include <boost/mpi/collectives.hpp>
//#include <boost/mpi/communicator.hpp>
//#include <boost/mpi/environment.hpp>
#include <mpi.h>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include "core/task/include/task.hpp"

namespace kabalova_v_my_reduce {

template <typename T>
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<T> input, local_input_;
  T result{};
  MPI_Comm comm;
  std::string ops;
};

}  // namespace kabalova_v_my_reduce