// Copyright 2024 Kabalova Valeria
#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kabalova_v_count_symbols/include/count_symbols.hpp"

TEST(kabalova_v_count_symbols_seq_perf_test, test_pipeline_run) {
  std::string string = "War and Peace is a literary work by the Russian author Lev Tolstoy. \
                     Set during the Napoleonic Wars, the work comprises both a fictional \
                     narrative and chapters in which Tolstoy discusses history and philosophy.\
                     An early version was published serially beginning in 1865, after which the \
                     entire book was rewritten and published in 1869. It is regarded, with Anna Karenina,\
                     as Tolstoy's finest literary achievement, and it remains an internationally praised classic \
                     of world literature.";
  std::string str; 
  for (int i = 0; i < 200; i++) {
    str += string;
  }
  // Create data
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kabalova_v_count_symbols_seq::Task1Seq>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(kabalova_v_count_symbols_seq_perf_test, test_task_run) {
  std::string string = "War and Peace is a literary work by the Russian author Lev Tolstoy. \
                     Set during the Napoleonic Wars, the work comprises both a fictional \
                     narrative and chapters in which Tolstoy discusses history and philosophy.\
                     An early version was published serially beginning in 1865, after which the \
                     entire book was rewritten and published in 1869. It is regarded, with Anna Karenina,\
                     as Tolstoy's finest literary achievement, and it remains an internationally praised classic \
                     of world literature.";
  std::string str;
  for (int i = 0; i < 200; i++) {
    str += string;
  }

  // Create data
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kabalova_v_count_symbols_seq::Task1Seq>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}