// Copyright 2024 Ivanov Mike
#include <gtest/gtest.h>

#include "seq/ivanov_m_optimization_by_characteristics/include/ops_seq.hpp"

TEST(ivanov_m_optimization_by_characteristics_seq_func_test, validation_true) {
  // start information
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 11;                // size in points
  double step = 1;              // length of step
  double approximation = 1e-2;  // approximation of the result

  // vector of start information
  std::vector<double> info{centerX, centerY, static_cast<double>(size), step, approximation};

  // main function
  std::function<double(double, double)> f = [](double x, double y) { return x + y; };

  // restriction functions
  std::function<bool(double, double)> r1 = [](double x, double y) {
    (void)x;
    return y > 1;
  };
  std::vector<std::function<bool(double, double)>> restriction{r1};

  // result
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
  taskDataSeq->inputs_count.emplace_back(info.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task

  ivanov_m_optimization_by_characteristics_seq::TestTaskSequential testTaskSequential(taskDataSeq, f, restriction);

  EXPECT_EQ(testTaskSequential.validation(), true);
}

TEST(ivanov_m_optimization_by_characteristics_seq_func_test, validation_false_1) {
  // start information
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 11;                // size in points
  double step = 1;              // length of step
  double approximation = 1e-2;  // approximation of the result

  // vector of start information
  std::vector<double> info{centerX, centerY, static_cast<double>(size), step, approximation};
  std::vector<double> info_false{centerX, centerY};

  // main function
  std::function<double(double, double)> f = [](double x, double y) { return x + y; };

  // restriction functions
  std::function<bool(double, double)> r1 = [](double x, double y) {
    (void)x;
    return y > 1;
  };
  std::vector<std::function<bool(double, double)>> restriction{r1};

  // result
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
  taskDataSeq->inputs_count.emplace_back(info.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info_false.data()));
  taskDataSeq->inputs_count.emplace_back(info_false.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task

  ivanov_m_optimization_by_characteristics_seq::TestTaskSequential testTaskSequential(taskDataSeq, f, restriction);

  EXPECT_EQ(testTaskSequential.validation(), false);
}

TEST(ivanov_m_optimization_by_characteristics_seq_func_test, validation_false_2) {
  // start information
  double centerX = 0;  // coordinate X of the search area center
  double centerY = 0;  // coordinate Y of the search area center
  int size = 11;       // size in points

  // vector of start information
  std::vector<double> info{centerX, centerY, static_cast<double>(size)};

  // main function
  std::function<double(double, double)> f = [](double x, double y) { return x + y; };

  // restriction functions
  std::function<bool(double, double)> r1 = [](double x, double y) {
    (void)x;
    return y > 1;
  };
  std::vector<std::function<bool(double, double)>> restriction{r1};

  // result
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
  taskDataSeq->inputs_count.emplace_back(info.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task

  ivanov_m_optimization_by_characteristics_seq::TestTaskSequential testTaskSequential(taskDataSeq, f, restriction);

  EXPECT_EQ(testTaskSequential.validation(), false);
}

TEST(ivanov_m_optimization_by_characteristics_seq_func_test, validation_false_3) {
  // start information
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 11;                // size in points
  double step = 1;              // length of step
  double approximation = 1e-2;  // approximation of the result

  // vector of start information
  std::vector<double> info{centerX, centerY, static_cast<double>(size), step, approximation};

  // main function
  std::function<double(double, double)> f = [](double x, double y) { return x + y; };

  // restriction functions
  std::function<bool(double, double)> r1 = [](double x, double y) {
    (void)x;
    return y > 1;
  };
  std::vector<std::function<bool(double, double)>> restriction{r1};

  // result
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
  taskDataSeq->inputs_count.emplace_back(info.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(3);

  // Create Task

  ivanov_m_optimization_by_characteristics_seq::TestTaskSequential testTaskSequential(taskDataSeq, f, restriction);

  EXPECT_EQ(testTaskSequential.validation(), false);
}

TEST(ivanov_m_optimization_by_characteristics_seq_func_test, pre_processing) {
  // start information
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 11;                // size in points
  double step = 1;              // length of step
  double approximation = 1e-2;  // approximation of the result

  // vector of start information
  std::vector<double> info{centerX, centerY, static_cast<double>(size), step, approximation};

  // main function
  std::function<double(double, double)> f = [](double x, double y) { return x + y; };

  // restriction functions
  std::function<bool(double, double)> r1 = [](double x, double y) {
    (void)x;
    return y > 1;
  };
  std::vector<std::function<bool(double, double)>> restriction{r1};

  // result
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
  taskDataSeq->inputs_count.emplace_back(info.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task

  ivanov_m_optimization_by_characteristics_seq::TestTaskSequential testTaskSequential(taskDataSeq, f, restriction);

  ASSERT_TRUE(testTaskSequential.validation());
  EXPECT_EQ(testTaskSequential.pre_processing(), true);
}

TEST(ivanov_m_optimization_by_characteristics_seq_func_test, run_simple_test_1_restriction_1) {
  // start information
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 11;                // size in points
  double step = 1;              // length of step
  double approximation = 1e-3;  // approximation of the result

  // vector of start information
  std::vector<double> info{centerX, centerY, static_cast<double>(size), step, approximation};

  // main function
  std::function<double(double, double)> f = [](double x, double y) { return x * x + y * y; };

  // restriction functions
  std::function<bool(double, double)> r1 = [](double x, double y) {
    (void)x;
    return y > 1;
  };
  std::vector<std::function<bool(double, double)>> restriction{r1};

  // result
  double res = 1;
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
  taskDataSeq->inputs_count.emplace_back(info.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task

  ivanov_m_optimization_by_characteristics_seq::TestTaskSequential testTaskSequential(taskDataSeq, f, restriction);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  EXPECT_NEAR(res, out, 1e-3);
}

TEST(ivanov_m_optimization_by_characteristics_seq_func_test, run_simple_test_1_restriction_2) {
  // start information
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 11;                // size in points
  double step = 1;              // length of step
  double approximation = 1e-3;  // approximation of the result

  // vector of start information
  std::vector<double> info{centerX, centerY, static_cast<double>(size), step, approximation};

  // main function
  std::function<double(double, double)> f = [](double x, double y) { return x * x + y * y; };

  // restriction functions
  std::function<bool(double, double)> r1 = [](double x, double y) {
    (void)x;
    return y > 1;
  };
  std::vector<std::function<bool(double, double)>> restriction{r1};

  // result
  double res = 1;
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
  taskDataSeq->inputs_count.emplace_back(info.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task

  ivanov_m_optimization_by_characteristics_seq::TestTaskSequential testTaskSequential(taskDataSeq, f, restriction);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  EXPECT_NEAR(res, out, 1e-3);
}

TEST(ivanov_m_optimization_by_characteristics_seq_func_test, run_simple_test_1_restriction_3) {
  // start information
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 101;               // size in points
  double step = 0.1;            // length of step
  double approximation = 1e-3;  // approximation of the result

  // vector of start information
  std::vector<double> info{centerX, centerY, static_cast<double>(size), step, approximation};

  // main function
  std::function<double(double, double)> f = [](double x, double y) { return x * x + y * y; };

  // restriction functions
  std::function<bool(double, double)> r1 = [](double x, double y) {
    (void)x;
    return y > 1;
  };
  std::function<bool(double, double)> r2 = [](double x, double y) { return y < x; };
  std::function<bool(double, double)> r3 = [](double x, double y) { return 4 > x * x + y * y; };
  std::vector<std::function<bool(double, double)>> restriction{r1, r2, r3};

  // result
  double res = 2;
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
  taskDataSeq->inputs_count.emplace_back(info.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task

  ivanov_m_optimization_by_characteristics_seq::TestTaskSequential testTaskSequential(taskDataSeq, f, restriction);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  EXPECT_NEAR(res, out, 1e-3);
}

TEST(ivanov_m_optimization_by_characteristics_seq_func_test, run_simple_test_1_restriction_3_negative) {
  // start information
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 11;                // size in points
  double step = 1;              // length of step
  double approximation = 1e-3;  // approximation of the result

  // vector of start information
  std::vector<double> info{centerX, centerY, static_cast<double>(size), step, approximation};

  // main function
  std::function<double(double, double)> f = [](double x, double y) { return x * x + y * y; };

  // restriction functions
  std::function<bool(double, double)> r1 = [](double x, double y) {
    (void)x;
    return y > 1;
  };
  std::function<bool(double, double)> r2 = [](double x, double y) { return y < x; };
  std::function<bool(double, double)> r3 = [](double x, double y) { return 4 > x * x + y * y; };
  std::vector<std::function<bool(double, double)>> restriction{r1, r2, r3};

  // result
  double res = INT_MAX;
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
  taskDataSeq->inputs_count.emplace_back(info.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task

  ivanov_m_optimization_by_characteristics_seq::TestTaskSequential testTaskSequential(taskDataSeq, f, restriction);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  EXPECT_NEAR(res, out, 1);
}

TEST(ivanov_m_optimization_by_characteristics_seq_func_test, run_hard_test_1_restriction_3) {
  // start information (area 5x5 with center (0, 0))
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 101;               // size in points
  double step = 0.1;            // length of step
  double approximation = 1e-6;  // approximation of the result

  // vector of start information
  std::vector<double> info{centerX, centerY, static_cast<double>(size), step, approximation};

  // main function
  std::function<double(double, double)> f = [](double x, double y) { return y * log(x); };

  // restriction functions
  std::function<bool(double, double)> r1 = [](double x, double y) {
    (void)y;
    return x > 1;
  };
  std::function<bool(double, double)> r2 = [](double x, double y) { return std::numbers::e <= sqrt(x * x + y * y); };
  std::function<bool(double, double)> r3 = [](double x, double y) { return 1 > (x - 2) * (x - 2) + (y - 1) * (y - 1); };

  std::vector<std::function<bool(double, double)>> restriction{r1, r2, r3};

  // result
  double res = f(2.71, 0.29949);
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
  taskDataSeq->inputs_count.emplace_back(info.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task

  ivanov_m_optimization_by_characteristics_seq::TestTaskSequential testTaskSequential(taskDataSeq, f, restriction);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  EXPECT_NEAR(res, out, 1e-2);
}

TEST(ivanov_m_optimization_by_characteristics_seq_func_test, run_hard_test_2_restriction_8) {
  // start information (area 5x5 with center (0, 0))
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 101;               // size in points
  double step = 0.1;            // length of step
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
  double res = 25;
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
  taskDataSeq->inputs_count.emplace_back(info.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task

  ivanov_m_optimization_by_characteristics_seq::TestTaskSequential testTaskSequential(taskDataSeq, f, restriction);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  EXPECT_NEAR(res, out, 1e-3);
}