// Copyright 2024 Ivanov Mike
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/ivanov_m_optimization_by_characteristics/include/ops_mpi.hpp"

TEST(ivanov_m_optimization_by_characteristics_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  // start information (area 5x5 with center (0, 0))
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 401;               // size in points
  double step = 0.025;          // length of step
  double approximation = 1e-6;  // approximation of the result

  // vector of start information
  std::vector<double> info{centerX, centerY, static_cast<double>(size), step, approximation};

  // main function
  std::function<double(double, double)> f = [](double x, double y) { return x * x + y * y; };

  // restriction functions
  std::function<bool(double, double)> r1 = [](double x, double y) { return 25 < x * x + y * y; };
  std::function<bool(double, double)> r2 = [](double x, double y) { return x < sqrt(y + 5); };
  std::function<bool(double, double)> r3 = [](double x, double y) { return y > (-1) * log(x) - 5; };
  std::function<bool(double, double)> r4 = [](double x, double y) {
    (void)x;
    return 1 < abs(y);
  };
  std::function<bool(double, double)> r5 = [](double x, double y) { return y > 5 * sin(x); };
  std::function<bool(double, double)> r6 = [](double x, double y) { return y > x + 1; };
  std::function<bool(double, double)> r7 = [](double x, double y) {
    (void)x;
    return y > 4;
  };
  std::function<bool(double, double)> r8 = [](double x, double y) { return x < y * y; };

  std::vector<std::function<bool(double, double)>> restriction{r1, r2, r3, r4, r5, r6, r7, r8};

  // result
  double out_mpi = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataPar->inputs_count.emplace_back(info.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  // Create Task
  auto task =
      std::make_shared<ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel>(taskDataPar, f, restriction);

  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(ivanov_m_optimization_by_characteristics_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;
  // start information (area 5x5 with center (0, 0))
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 401;               // size in points
  double step = 0.025;          // length of step
  double approximation = 1e-6;  // approximation of the result

  // vector of start information
  std::vector<double> info{centerX, centerY, static_cast<double>(size), step, approximation};

  // main function
  std::function<double(double, double)> f = [](double x, double y) { return x * x + y * y; };

  // restriction functions
  std::function<bool(double, double)> r1 = [](double x, double y) { return 25 < x * x + y * y; };
  std::function<bool(double, double)> r2 = [](double x, double y) { return x < sqrt(y + 5); };
  std::function<bool(double, double)> r3 = [](double x, double y) { return y > (-1) * log(x) - 5; };
  std::function<bool(double, double)> r4 = [](double x, double y) {
    (void)x;
    return 1 < abs(y);
  };
  std::function<bool(double, double)> r5 = [](double x, double y) { return y > 5 * sin(x); };
  std::function<bool(double, double)> r6 = [](double x, double y) { return y > x + 1; };
  std::function<bool(double, double)> r7 = [](double x, double y) {
    (void)x;
    return y > 4;
  };
  std::function<bool(double, double)> r8 = [](double x, double y) { return x < y * y; };

  std::vector<std::function<bool(double, double)>> restriction{r1, r2, r3, r4, r5, r6, r7, r8};

  // result
  double out_mpi = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataPar->inputs_count.emplace_back(info.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  // Create Task
  auto task =
      std::make_shared<ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel>(taskDataPar, f, restriction);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}