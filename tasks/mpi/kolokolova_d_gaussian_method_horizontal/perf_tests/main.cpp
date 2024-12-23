// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/timer.hpp>
#include <functional>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kolokolova_d_gaussian_method_horizontal/include/ops_mpi.hpp"

using namespace kolokolova_d_gaussian_method_horizontal_mpi;

std::vector<int> kolokolova_d_gaussian_method_horizontal_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  std::uniform_int_distribution<int> dist(1, 100);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

std::vector<int> kolokolova_d_gaussian_method_horizontal_mpi::getRandomMatrix(int sz, std::vector<int> coeff) {
  std::vector<int> free_terms(sz);
  for (int i = 0; i < sz; ++i) {
    free_terms[i] = 0;
    for (int j = 0; j < sz; ++j) {
      free_terms[i] += coeff[i * sz + j];
    }
  }
  return free_terms;
}

TEST(kolokolova_d_gaussian_method_horizontal_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int count_equations = 240;
  int size_coef_mat = count_equations * count_equations;
  std::vector<int> input_coeff(size_coef_mat, 0);
  std::vector<int> input_y(count_equations, 0);
  std::vector<double> func_res(count_equations, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_coeff = kolokolova_d_gaussian_method_horizontal_mpi::getRandomVector(size_coef_mat);
    input_y = kolokolova_d_gaussian_method_horizontal_mpi::getRandomMatrix(count_equations, input_coeff);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
    taskDataPar->inputs_count.emplace_back(input_coeff.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
    taskDataPar->inputs_count.emplace_back(input_y.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
    taskDataPar->outputs_count.emplace_back(func_res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(int(func_res.size()), count_equations);
  }
}

TEST(kolokolova_d_gaussian_method_horizontal_mpi, test_task_run) {
  boost::mpi::communicator world;
  int count_equations = 240;
  int size_coef_mat = count_equations * count_equations;
  std::vector<int> input_coeff(size_coef_mat, 0);
  std::vector<int> input_y(count_equations, 0);
  std::vector<double> func_res(count_equations, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_coeff = kolokolova_d_gaussian_method_horizontal_mpi::getRandomVector(size_coef_mat);
    input_y = kolokolova_d_gaussian_method_horizontal_mpi::getRandomMatrix(count_equations, input_coeff);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
    taskDataPar->inputs_count.emplace_back(input_coeff.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
    taskDataPar->inputs_count.emplace_back(input_y.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
    taskDataPar->outputs_count.emplace_back(func_res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(int(func_res.size()), count_equations);
  }
}