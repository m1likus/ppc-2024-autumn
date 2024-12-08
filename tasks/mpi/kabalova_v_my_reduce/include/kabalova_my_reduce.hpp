// Copyright 2024 Kabalova Valeria
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/operations.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/packed_iarchive.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kabalova_v_my_reduce {
bool checkValidOperation(std::string operation);
void myReduce(const boost::mpi::communicator& comm, const int& inValue, int& outValue, std::string operation, int root);
void reduceImplementation(const boost::mpi::communicator& comm, const int& inValue, int& outValue, std::string operation, int root);
void reduceImplementation(const boost::mpi::communicator& comm, const int& inValue, std::string operation, int root);
void reduceTree(const boost::mpi::communicator &comm, const int& inValue, int& outValue, std::string operation, int root);
void reduceTree(const boost::mpi::communicator &comm, const int& inValue, std::string operation, int root);
int op(const int& a, const int& b, std::string operation);

class Tree {
 private:
  int rank;
  int size;
  int root;
  int level_;
 public:
  Tree(int rank, int size, int root);

  // Level in the tree, where the proccess is right now
  int level() const { 
    return level_;
  }
  // On what layer we sit, nth level of the tree
  int levelIndex(int n) const;
  int parent() const;
  // First child of the current process
  int begin() const;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_{}, local_input_{};
  int result{};
  boost::mpi::communicator world;
  std::string ops;
};

}  // namespace kabalova_v_my_reduce