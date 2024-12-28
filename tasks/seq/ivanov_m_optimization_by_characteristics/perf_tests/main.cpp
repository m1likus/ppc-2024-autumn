// Copyright 2024 Ivanov Mike
#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/ivanov_m_optimization_by_characteristics/include/ops_seq.hpp"

TEST(ivanov_m_optimization_by_characteristics_seq_perf_test, test_pipeline_run) {
  // start information (area 5x5 with center (0, 0))
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 5001;              // size in points
  double step = 0.002;          // length of step
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

  // result vector
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
  taskDataSeq->inputs_count.emplace_back(info.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  auto task =
      std::make_shared<ivanov_m_optimization_by_characteristics_seq::TestTaskSequential>(taskDataSeq, f, restriction);

  // create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(ivanov_m_optimization_by_characteristics_seq_perf_test, test_task_run) {
  // start information (area 5x5 with center (0, 0))
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 5001;              // size in points
  double step = 0.002;          // length of step
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

  // result vector
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
  taskDataSeq->inputs_count.emplace_back(info.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  auto task =
      std::make_shared<ivanov_m_optimization_by_characteristics_seq::TestTaskSequential>(taskDataSeq, f, restriction);

  // create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}