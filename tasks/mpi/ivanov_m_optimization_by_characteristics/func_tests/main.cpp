// Copyright 2024 Ivanov Mike
#include <gtest/gtest.h>

#include "mpi/ivanov_m_optimization_by_characteristics/include/ops_mpi.hpp"

TEST(ivanov_m_optimization_by_characteristics_mpi_func_test, validation) {
  boost::mpi::communicator world;
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
    return y >= 1;
  };
  std::vector<std::function<bool(double, double)>> restriction{r1};

  // result
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataPar->inputs_count.emplace_back(info.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
  }

  if (world.rank() == 0) {
    ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f, restriction);

    EXPECT_EQ(testMpiTaskParallel.validation(), true);
  }
}

TEST(ivanov_m_optimization_by_characteristics_mpi_func_test, validation_false_1) {
  boost::mpi::communicator world;
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
    return y >= 1;
  };
  std::vector<std::function<bool(double, double)>> restriction{r1};

  // result
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataPar->inputs_count.emplace_back(info.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info_false.data()));
    taskDataPar->inputs_count.emplace_back(info_false.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
  }

  if (world.rank() == 0) {
    ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f, restriction);

    EXPECT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(ivanov_m_optimization_by_characteristics_mpi_func_test, validation_false_2) {
  boost::mpi::communicator world;
  // start information
  double centerX = 0;  // coordinate X of the search area center
  double centerY = 0;  // coordinate Y of the search area center

  // vector of start information
  std::vector<double> info{centerX, centerY};

  // main function
  std::function<double(double, double)> f = [](double x, double y) { return x + y; };

  // restriction functions
  std::function<bool(double, double)> r1 = [](double x, double y) {
    (void)x;
    return y >= 1;
  };
  std::vector<std::function<bool(double, double)>> restriction{r1};

  // result
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataPar->inputs_count.emplace_back(info.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
  }

  if (world.rank() == 0) {
    ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f, restriction);

    EXPECT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(ivanov_m_optimization_by_characteristics_mpi_func_test, validation_false_3) {
  boost::mpi::communicator world;
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
    return y >= 1;
  };
  std::vector<std::function<bool(double, double)>> restriction{r1};

  // result
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataPar->inputs_count.emplace_back(info.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(3);
  }

  if (world.rank() == 0) {
    ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f, restriction);

    EXPECT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(ivanov_m_optimization_by_characteristics_mpi_func_test, pre_processing) {
  boost::mpi::communicator world;
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
    return y >= 1;
  };
  std::vector<std::function<bool(double, double)>> restriction{r1};

  // result
  double out = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataPar->inputs_count.emplace_back(info.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
  }

  if (world.rank() == 0) {
    ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f, restriction);

    ASSERT_TRUE(testMpiTaskParallel.validation());
    EXPECT_EQ(testMpiTaskParallel.pre_processing(), true);
  }
}

TEST(ivanov_m_optimization_by_characteristics_mpi_func_test, run_simple_test_1_restriction_1) {
  boost::mpi::communicator world;
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
    return y >= 1;
  };
  std::vector<std::function<bool(double, double)>> restriction{r1};

  // result (correct: 1)
  double out_seq = INT_MAX;
  double out_mpi = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataPar->inputs_count.emplace_back(info.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f, restriction);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataSeq->inputs_count.emplace_back(info.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task

    ivanov_m_optimization_by_characteristics_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f,
                                                                                              restriction);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    EXPECT_NEAR(out_mpi, out_seq, 1e-2);
  }
}

TEST(ivanov_m_optimization_by_characteristics_mpi_func_test, run_simple_test_1_restriction_2) {
  boost::mpi::communicator world;
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

  // result (correct: 1)
  double out_seq = INT_MAX;
  double out_mpi = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataPar->inputs_count.emplace_back(info.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f, restriction);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataSeq->inputs_count.emplace_back(info.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task

    ivanov_m_optimization_by_characteristics_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f,
                                                                                              restriction);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    EXPECT_NEAR(out_mpi, out_seq, 1e-2);
  }
}

TEST(ivanov_m_optimization_by_characteristics_mpi_func_test, run_simple_test_1_restriction_3) {
  boost::mpi::communicator world;
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

  // result (correct: 2)
  double out_seq = INT_MAX;
  double out_mpi = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataPar->inputs_count.emplace_back(info.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f, restriction);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataSeq->inputs_count.emplace_back(info.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task

    ivanov_m_optimization_by_characteristics_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f,
                                                                                              restriction);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    EXPECT_NEAR(out_mpi, out_seq, 1e-2);
  }
}

TEST(ivanov_m_optimization_by_characteristics_mpi_func_test, run_hard_test_1_restriction_3) {
  boost::mpi::communicator world;
  // start information (area 5x5 with center (0, 0))
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 11;                // size in points
  double step = 1;              // length of step
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
  std::function<bool(double, double)> r3 = [](double x, double y) { return 1 > (x - 3) * (x - 3) + (y - 1) * (y - 1); };

  std::vector<std::function<bool(double, double)>> restriction{r1, r2, r3};

  // result (correct: 2)
  double out_seq = INT_MAX;
  double out_mpi = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataPar->inputs_count.emplace_back(info.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f, restriction);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataSeq->inputs_count.emplace_back(info.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task

    ivanov_m_optimization_by_characteristics_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f,
                                                                                              restriction);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    EXPECT_NEAR(out_mpi, out_seq, 1e-2);
  }
}

TEST(ivanov_m_optimization_by_characteristics_mpi_func_test, run_hard_test_1_restriction_3_negative) {
  boost::mpi::communicator world;
  // start information (area 5x5 with center (0, 0))
  double centerX = 0;           // coordinate X of the search area center
  double centerY = 0;           // coordinate Y of the search area center
  int size = 11;                // size in points
  double step = 1;              // length of step
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

  // result (correct: 2)
  double out_seq = INT_MAX;
  double out_mpi = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataPar->inputs_count.emplace_back(info.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f, restriction);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataSeq->inputs_count.emplace_back(info.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task

    ivanov_m_optimization_by_characteristics_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f,
                                                                                              restriction);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    EXPECT_NEAR(out_mpi, out_seq, 1e-2);
    EXPECT_NEAR(out_mpi, INT_MAX, 1);
    EXPECT_NEAR(out_seq, INT_MAX, 1);
  }
}

TEST(ivanov_m_optimization_by_characteristics_mpi_func_test, run_hard_test_1_restriction_8) {
  boost::mpi::communicator world;
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

  // result (correct: 2)
  double out_seq = INT_MAX;
  double out_mpi = INT_MAX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataPar->inputs_count.emplace_back(info.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_mpi));
    taskDataPar->outputs_count.emplace_back(1);
  }

  ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f, restriction);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(info.data()));
    taskDataSeq->inputs_count.emplace_back(info.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task

    ivanov_m_optimization_by_characteristics_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f,
                                                                                              restriction);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    EXPECT_NEAR(out_mpi, out_seq, 1e-2);
  }
}
