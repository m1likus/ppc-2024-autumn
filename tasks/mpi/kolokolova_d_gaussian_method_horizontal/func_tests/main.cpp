#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <functional>
#include <random>
#include <vector>

#include "mpi/kolokolova_d_gaussian_method_horizontal/include/ops_mpi.hpp"

using namespace kolokolova_d_gaussian_method_horizontal_mpi;

std::vector<int> kolokolova_d_gaussian_method_horizontal_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  std::uniform_int_distribution<int> dist(-100, 100);
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

TEST(kolokolova_d_gaussian_method_horizontal_mpi, Test_Parallel_Gauss1) {
  // Test with nums
  boost::mpi::communicator world;
  int count_equations = 3;
  int size_coef_mat = count_equations * count_equations;
  std::vector<int> input_coeff = {3, 2, -5, 2, -1, 3, 1, 2, -1};
  std::vector<int> input_y = {-1, 13, 9};
  std::vector<double> func_res(count_equations, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_y = kolokolova_d_gaussian_method_horizontal_mpi::getRandomVector(count_equations);
    input_coeff = kolokolova_d_gaussian_method_horizontal_mpi::getRandomVector(size_coef_mat);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
    taskDataPar->inputs_count.emplace_back(input_coeff.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
    taskDataPar->inputs_count.emplace_back(input_y.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
    taskDataPar->outputs_count.emplace_back(func_res.size());
  }

  kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_res(count_equations, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
    taskDataSeq->inputs_count.emplace_back(input_coeff.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
    taskDataSeq->inputs_count.emplace_back(input_y.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < count_equations; i++) {
      ASSERT_EQ(func_res[i], reference_res[i]);
    }
  }
}

TEST(kolokolova_d_gaussian_method_horizontal_mpi, Test_Parallel_Gauss2) {
  // For each proc 10 equations
  boost::mpi::communicator world;
  int count_equations = world.size() * 10;
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

  kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_res(count_equations, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
    taskDataSeq->inputs_count.emplace_back(input_coeff.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
    taskDataSeq->inputs_count.emplace_back(input_y.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < count_equations; i++) {
      ASSERT_EQ(func_res[i], reference_res[i]);
    }
  }
}

TEST(kolokolova_d_gaussian_method_horizontal_mpi, Test_Parallel_Gauss3) {
  // For each proc 20 equations
  boost::mpi::communicator world;
  int count_equations = world.size() * 20;
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

  kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_res(count_equations, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
    taskDataSeq->inputs_count.emplace_back(input_coeff.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
    taskDataSeq->inputs_count.emplace_back(input_y.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < count_equations; i++) {
      ASSERT_EQ(func_res[i], reference_res[i]);
    }
  }
}

TEST(kolokolova_d_gaussian_method_horizontal_mpi, Test_Parallel_Gauss4) {
  // 100 equations
  boost::mpi::communicator world;
  int count_equations = 100;
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

  kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_res(count_equations, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
    taskDataSeq->inputs_count.emplace_back(input_coeff.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
    taskDataSeq->inputs_count.emplace_back(input_y.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < count_equations; i++) {
      ASSERT_EQ(func_res[i], reference_res[i]);
    }
  }
}

TEST(kolokolova_d_gaussian_method_horizontal_mpi, Test_Parallel_Gauss_Empyty_y) {
  // When input_y is empty
  boost::mpi::communicator world;
  int count_equations = world.size();
  int size_coef_mat = count_equations * count_equations;
  std::vector<int> input_coeff(size_coef_mat, 0);
  std::vector<int> input_y;
  std::vector<double> func_res(count_equations, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_coeff = kolokolova_d_gaussian_method_horizontal_mpi::getRandomVector(size_coef_mat);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
    taskDataPar->inputs_count.emplace_back(input_coeff.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
    taskDataPar->inputs_count.emplace_back(input_y.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
    taskDataPar->outputs_count.emplace_back(func_res.size());
  }

  if (world.rank() == 0) {
    kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kolokolova_d_gaussian_method_horizontal_mpi, Test_Parallel_Gauss_Empyty_coeff) {
  // When input_coeff is empty
  boost::mpi::communicator world;
  int count_equations = world.size();
  std::vector<int> input_coeff;
  std::vector<int> input_y(count_equations, 0);
  std::vector<double> func_res(count_equations, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_y = kolokolova_d_gaussian_method_horizontal_mpi::getRandomVector(count_equations);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
    taskDataPar->inputs_count.emplace_back(input_coeff.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
    taskDataPar->inputs_count.emplace_back(input_y.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
    taskDataPar->outputs_count.emplace_back(func_res.size());
  }

  if (world.rank() == 0) {
    kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kolokolova_d_gaussian_method_horizontal_mpi, Test_Parallel_rank) {
  // When main rank not equal full rank
  boost::mpi::communicator world;
  std::vector<int> input_coeff = {7, -2, -1, 6, -4, -5, 1, 2, 4};
  std::vector<int> input_y = {2, 3, 5};
  std::vector<double> func_res(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
    taskDataPar->inputs_count.emplace_back(input_coeff.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
    taskDataPar->inputs_count.emplace_back(input_y.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
    taskDataPar->outputs_count.emplace_back(func_res.size());
  }

  if (world.rank() == 0) {
    kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}