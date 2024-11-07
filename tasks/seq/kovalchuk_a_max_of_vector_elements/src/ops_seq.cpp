// Copyright 2023 Nesterov Alexander
#include "seq/kovalchuk_a_max_of_vector_elements/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

bool kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask::pre_processing() {
  internal_order_test();
  // Init vectors
  if (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0) {
    input_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], input_[i].begin());
    }
  } else {
    input_ = std::vector<std::vector<int>>();
  }
  // Init value for output
  res_ = INT_MIN;
  return true;
}

bool kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask::run() {
  internal_order_test();
  if (!input_.empty()) {
    std::vector<int> local_res(input_.size());
    for (unsigned int i = 0; i < input_.size(); i++) {
      if (!input_[i].empty()) {
        local_res[i] = *std::max_element(input_[i].begin(), input_[i].end());
      } else {
        local_res[i] = INT_MIN;
      }
    }
    res_ = *std::max_element(local_res.begin(), local_res.end());
  } else {
    res_ = INT_MIN;
  }
  return true;
}

bool kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}