// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/varfolomeev_g_matrix_max_rows_vals/include/ops_mpi.hpp"

TEST(mpi_varfolomeev_g_matrix_max_rows_perf_test, test_pipeline_run) {
  int size_m = 5000;
  int size_n = 5000;

  boost::mpi::communicator world;

  std::vector<int> matrix(size_n * size_m, 1);
  std::vector<int32_t> max_vec(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Setting rows(size_m) and cols(size_n)

  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);

  // If curr. proc. is root (r.0), setting the input and output data
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(max_vec.data()));
    taskDataPar->outputs_count.emplace_back(max_vec.size());
  }
  auto testMpiTaskParallel = std::make_shared<varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel>(taskDataPar);
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
  // If curr. proc. is root (r.0), display performance and check the result
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(1, max_vec[0]);
  }
}

TEST(mpi_varfolomeev_g_matrix_max_rows_perf_test, test_task_run) {
  int size_m = 5000;
  int size_n = 5000;
  boost::mpi::communicator world;
  std::vector<int> matrix(size_n * size_m, 1);
  std::vector<int32_t> max_vec(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(max_vec.data()));
    taskDataPar->outputs_count.emplace_back(max_vec.size());
  }
  auto testMpiTaskParallel = std::make_shared<varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel>(taskDataPar);
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
    ASSERT_EQ(1, max_vec[0]);
  }
}