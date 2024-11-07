// Copyright 2024 Sedova Olga
#include "seq/sedova_o_max_of_vector_elements/include/ops_seq.hpp"

int sedova_o_max_of_vector_elements_seq::find_max_of_matrix(std::vector<int> matrix) {
  if (matrix.empty()) return 1;
  int max = matrix[0];
  for (size_t i = 0; i < matrix.size(); i++) {
    if (matrix[i] > max) {
      max = matrix[i];
    }
  }
  return max;
}

bool sedova_o_max_of_vector_elements_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  unsigned int rows = taskData->inputs_count[0];
  unsigned int cols = taskData->inputs_count[1];
  input_ = std::vector<int>(rows * cols);
  for (unsigned int i = 0; i < rows; i++) {
    auto* input_data = reinterpret_cast<int*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i * cols + j] = input_data[j];
    }
  }
  return true;
}

bool sedova_o_max_of_vector_elements_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 1 && taskData->inputs_count[1] >= 1 && taskData->outputs_count[0] == 1;
}

bool sedova_o_max_of_vector_elements_seq::TestTaskSequential::run() {
  internal_order_test();
  res_ = sedova_o_max_of_vector_elements_seq::find_max_of_matrix(input_);
  return true;
}

bool sedova_o_max_of_vector_elements_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}