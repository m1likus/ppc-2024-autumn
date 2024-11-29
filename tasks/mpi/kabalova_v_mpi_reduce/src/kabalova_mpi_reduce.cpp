// Copyright 2024 Kabalova Valeria
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "mpi/kabalova_v_mpi_reduce/include/kabalova_mpi_reduce.hpp"

// List of valid operations:
// + = MPI_SUM
// * = MPI_PROD
// min = MPI_MIN
// max = MPI_MAX
// && = MPI_LAND
// || = MPI_LOR
// & = MPI_BAND
// | = MPI_BOR
// ^ = MPI_BXOR
// lxor = MPI_LXOR

bool kabalova_v_mpi_reduce::checkValidOperation(std::string ops) { 
  if (ops == "+" || ops == "*" || ops == "min" || ops == "max" || ops == "&&" || ops == "||" || ops == "&" ||
      ops == "|" || ops == "^" || ops == "lxor")
    return true;
  else
    return false;
}

bool kabalova_v_mpi_reduce::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init value for input
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    // Value for output is already initialized with result{}
  }
  return true;
}

bool kabalova_v_mpi_reduce::TestMPITaskParallel::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1 &&
          checkValidOperation(ops));
}

bool kabalova_v_mpi_reduce::TestMPITaskParallel::run() {
  internal_order_test();
  // Starting with proper partition of vector
  unsigned int delta = 0;
  if (world.rank() == 0) {
    // Get delta = vec.size() / num_threads
    delta = taskData->inputs_count[0] % world.size() == 0 ? taskData->inputs_count[0] / world.size()
                                                          : taskData->inputs_count[0] / world.size() + 1;
  }
  // Broadcast delta to every process that we have
  broadcast(world, delta, 0);
  // So we already have input_ because of pre_processing
  // Now we need to send subvectors properly
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      int bufDelta = 0;
      if ((size_t)(proc * delta + delta) > input_.size() && (size_t)proc < input_.size()) {
        bufDelta = input_.size() - proc * delta - delta;
      }
      world.send(proc, 0, input_.data() + proc * delta, delta + bufDelta);
    }
  }
  local_input_ = std::vector<int>(delta);
  if (world.rank() == 0) 
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  else {
    //std::vector<int> buffer = std::vector<int>(delta);
    world.recv(0, 0, local_input_.data(), delta);
    //local_input_ = std::vector<int>(buffer.begin(), buffer.begin() + delta);
  }

  // After this we can finally use reduce
  int local_res;
  if (ops == "+") { // MPI_SUM
    local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
    reduce(world, local_res, result, std::plus(), 0);
  } else if (ops == "*") { // MPI_PROD
    local_res = 1;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res *= local_input_[i];
    }
    reduce(world, local_res, result, std::multiplies(), 0);
  } else if (ops == "max") { // MPI_MAX
    local_res = *std::max_element(local_input_.begin(), local_input_.end());
    reduce(world, local_res, result, boost::mpi::maximum<int>(), 0);
  } else if (ops == "min") { // MPI_MIN
    local_res = *std::min_element(local_input_.begin(), local_input_.end());
    reduce(world, local_res, result, boost::mpi::minimum<int>(), 0);
  } else if (ops == "&&") { //MPI_LAND
    local_res = 1;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = local_res && local_input_[i];
    }
    reduce(world, local_res, result, std::logical_and(), 0);
  } else if (ops == "||") { //MPI_LOR
    local_res = 0;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = local_res || local_input_[i];
    }
    reduce(world, local_res, result, std::logical_or(), 0);
  } else if (ops == "&") {  // MPI_BAND
    local_res = 1;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = local_res & local_input_[i];
    }
    reduce(world, local_res, result, boost::mpi::bitwise_and<int>(), 0);
  } else if (ops == "|") {  // MPI_BOR
    local_res = 0;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = local_res | local_input_[i];
    }
    reduce(world, local_res, result, boost::mpi::bitwise_or<int>(), 0);
  } else if (ops == "^") { // MPI_BXOR
    local_res = 0;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = local_res ^ local_input_[i];
    }
    reduce(world, local_res, result, boost::mpi::bitwise_xor<int>(), 0);
  } else if (ops == "lxor") {  // MPI_LXOR
    local_res = 0;
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_res = !local_res != !local_input_[i];
    }
    reduce(world, local_res, result, boost::mpi::logical_xor<int>(), 0);
  }
  return true;
}

bool kabalova_v_mpi_reduce::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  }
  return true;
}