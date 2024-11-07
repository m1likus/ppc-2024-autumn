﻿// Copyright 2024 Sedova Olga
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/sedova_o_max_of_vector_elements/include/ops_seq.hpp"

namespace sedova_o_max_of_vector_elements_seq_test {

std::vector<int> generate_random_vector(size_t size, size_t value) {
  std::random_device dev;
  std::mt19937 random(dev());
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = random() % (value + 1);
  }
  return vec;
}

std::vector<std::vector<int>> generate_random_matrix(size_t rows, size_t cols, size_t value) {
  std::vector<std::vector<int>> matrix(rows);
  for (size_t i = 0; i < rows; i++) {
    matrix[i] = generate_random_vector(cols, value);
  }
  return matrix;
}
}  // namespace sedova_o_max_of_vector_elements_seq_test

TEST(sedova_o_max_of_vector_elements_seq1, Test_Sum_Empty1) {
  // Create data
  std::vector<int> in;
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  sedova_o_max_of_vector_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(sedova_o_max_of_vector_elements_seq1, Test_Sum_Input_Incorrect) {
  int count = 10;
  // Create data
  std::vector<int> in(count, 0);
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(0);  // Неверный размер входного вектора
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  sedova_o_max_of_vector_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(sedova_o_max_of_vector_elements_seq1, Test_Matrix_2x2) {
  // Create data
  std::vector<int> in = sedova_o_max_of_vector_elements_seq_test::generate_random_vector(2, 10);
  std::vector<int> in2 = sedova_o_max_of_vector_elements_seq_test::generate_random_vector(2, 10);
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  sedova_o_max_of_vector_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<int> matrix_input = {in[0], in[1], in2[0], in2[1]};
  ASSERT_EQ(sedova_o_max_of_vector_elements_seq::find_max_of_matrix(matrix_input), out[0]);
}

TEST(sedova_o_max_of_vector_elements_seq1, Test_Matrix_3x3) {
  // Create data
  std::vector<int> in = sedova_o_max_of_vector_elements_seq_test::generate_random_vector(3, 10);
  std::vector<int> in2 = sedova_o_max_of_vector_elements_seq_test::generate_random_vector(3, 10);
  std::vector<int> in3 = sedova_o_max_of_vector_elements_seq_test::generate_random_vector(3, 10);
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->inputs_count.emplace_back(in3.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  sedova_o_max_of_vector_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<int> matrix_input = {in[0], in[1], in[2], in2[0], in2[1], in2[2], in3[0], in3[1], in3[2]};
  ASSERT_EQ(sedova_o_max_of_vector_elements_seq::find_max_of_matrix(matrix_input), out[0]);
}
