// Copyright 2024 Ivanov Mike
#include "mpi/ivanov_m_optimization_by_characteristics/include/ops_mpi.hpp"

bool ivanov_m_optimization_by_characteristics_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input
  centerX = reinterpret_cast<double*>(taskData->inputs[0])[0];
  centerY = reinterpret_cast<double*>(taskData->inputs[0])[1];
  size = static_cast<int>(reinterpret_cast<double*>(taskData->inputs[0])[2]);
  step = reinterpret_cast<double*>(taskData->inputs[0])[3];
  approximation = reinterpret_cast<double*>(taskData->inputs[0])[4];

  // Init value for ouput
  res = *reinterpret_cast<double*>(taskData->outputs[0]);

  return (size > 1 && step > 0.0 && approximation > 0.0);
}

bool ivanov_m_optimization_by_characteristics_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return (static_cast<int>(taskData->inputs.size()) == 1 && static_cast<int>(taskData->inputs_count.size()) == 1 &&
          static_cast<int>(taskData->inputs_count[0]) == 5 && static_cast<int>(taskData->outputs.size()) == 1 &&
          static_cast<int>(taskData->outputs_count.size()) == 1 && static_cast<int>(taskData->outputs_count[0]) == 1);
}

bool ivanov_m_optimization_by_characteristics_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  // during algorithm we will move from left to right and from up to down
  double minX = centerX - step * (static_cast<double>(size - 1) / 2);  // minimal X of area
  double maxY = centerY + step * (static_cast<double>(size - 1) / 2);  // maximal Y of area

  double localX;  // coordinate X of local point
  double localY;  // coordinate X of local point

  double local_res = res - 2 * approximation;  // local res on current step
  double resX = centerX;                       // coordinate Y of local res
  double resY = centerY;                       // coordinate Y of local res

  int amount_of_restrictions = restriction.size();
  bool is_appropriate;  // flag of point appropriation
  int i;                // number of current restriction
  double local_value;   // value of function in point (localX, localY)

  while (std::abs(local_res - res) > approximation) {
    res = local_res;
    localY = maxY;
    for (int counterY = 0; counterY < size; counterY++) {
      localX = minX;
      for (int counterX = 0; counterX < size; counterX++) {
        is_appropriate = true;
        i = 0;
        while (i < amount_of_restrictions && is_appropriate) {
          is_appropriate = restriction[i](localX, localY);
          i++;
        }
        if (is_appropriate) {
          local_value = f(localX, localY);
          if (local_value < local_res) {
            resX = localX;
            resY = localY;
            local_res = local_value;
          }
        }
        localX += step;
      }
      localY -= step;
    }
    // updating area
    centerX = resX;
    centerY = resY;
    step = 2 * step / static_cast<double>(size - 1);
    minX = centerX - step * (static_cast<double>(size - 1) / 2);
    maxY = centerY + step * (static_cast<double>(size - 1) / 2);
  }

  res = local_res;
  return true;
}

bool ivanov_m_optimization_by_characteristics_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init value for input
    centerX = reinterpret_cast<double*>(taskData->inputs[0])[0];
    centerY = reinterpret_cast<double*>(taskData->inputs[0])[1];
    size = static_cast<int>(reinterpret_cast<double*>(taskData->inputs[0])[2]);
    step = reinterpret_cast<double*>(taskData->inputs[0])[3];
    approximation = reinterpret_cast<double*>(taskData->inputs[0])[4];

    // Init value for ouput
    res = *reinterpret_cast<double*>(taskData->outputs[0]);

    return (size > 1 && step > 0.0 && approximation > 0.0);
  }
  return true;
}

bool ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (static_cast<int>(taskData->inputs.size()) == 1 && static_cast<int>(taskData->inputs_count.size()) == 1 &&
            static_cast<int>(taskData->inputs_count[0]) == 5 && static_cast<int>(taskData->outputs.size()) == 1 &&
            static_cast<int>(taskData->outputs_count.size()) == 1 && static_cast<int>(taskData->outputs_count[0]) == 1);
  }
  return true;
}

bool ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int world_size = world.size();  // amount of processes in communicator
  int rank = world.rank();        // rank of this process

  // sending necessary information to each process
  boost::mpi::broadcast(world, size, 0);
  boost::mpi::broadcast(world, step, 0);
  boost::mpi::broadcast(world, approximation, 0);

  // distribution of restrictions
  // START
  int restriction_size = static_cast<int>(restriction.size());  // amount of restrictions
  std::vector<int> rest_in_proc(world_size, 0);                 // storage of each process start index
  if (rank == 0) {
    // we temporary use rest_in_proc as a storage of each process amount of retrictions
    // initialization of amount of restrictions in each process
    int rest = restriction_size / world_size;
    if (rest > 0) {
      fill(rest_in_proc.begin(), rest_in_proc.end(), rest);
    }
    rest = restriction_size % world_size;
    if (rest > 0) {
      for (int i = 0; i < rest; i++) {
        rest_in_proc[i]++;
      }
    }
  }

  // storage of amount of restrictions for this process
  int local_restrictions_size;
  // sending to each process their amount of restrictions
  boost::mpi::scatter(world, rest_in_proc.data(), &local_restrictions_size, 1, 0);

  // now we rebuild vector rest_in_pros into a storage of first index
  int tmp = restriction_size;  // amount of unmarked restrictions
  for (int i = world_size - 1; i >= 0; i--) {
    rest_in_proc[i] = tmp - rest_in_proc[i];
    tmp = rest_in_proc[i];
  }

  // storage of this process start index
  int start_index;
  // sending to each process their amount of restrictions
  boost::mpi::scatter(world, rest_in_proc.data(), &start_index, 1, 0);
  // END

  double local_res = DBL_MAX;  // container for res for each process
  double prev_res = DBL_MAX;   // previous result
  double minX = INT_MIN;
  double maxY = INT_MAX;

  std::vector<double> info(4, 0);

  if (rank == 0) {
    local_res = res - 2 * approximation;
    prev_res = res;

    minX = centerX - step * (static_cast<double>(size - 1) / 2);  // minimal X of area
    maxY = centerY + step * (static_cast<double>(size - 1) / 2);  // maximal Y of area
    info = std::vector<double>{prev_res, local_res, minX, maxY};
  }

  // sending to each process
  boost::mpi::broadcast(world, info, 0);

  if (rank != 0) {
    prev_res = info[0];
    local_res = info[1];
    minX = info[2];
    maxY = info[3];
  }

  // variables for algorithm
  double localX;  // coordinate X of local point
  double localY;  // coordinate X of local point

  int is_appropriate = 1;        // flag of point appropriation
  int local_is_appropriate = 1;  // flag of point appropriation for each process
  double local_value;            // value of function in point (localX, localY)

  std::vector<int> flags(world_size, 0);

  // searching algorithm
  while (std::abs(local_res - prev_res) > approximation) {
    prev_res = local_res;
    localY = maxY;
    for (int counterY = 0; counterY < size; counterY++) {
      localX = minX;
      for (int counterX = 0; counterX < size; counterX++) {
        local_is_appropriate = 1;
        int i = start_index;
        while (i < start_index + local_restrictions_size && static_cast<bool>(local_is_appropriate)) {
          local_is_appropriate = static_cast<int>(restriction[i](localX, localY));
          i++;
        }
        boost::mpi::reduce(world, &local_is_appropriate, 1, &is_appropriate, std::logical_and<>(), 0);
        if (rank == 0 && static_cast<bool>(is_appropriate)) {
          local_value = f(localX, localY);
          if (local_value < local_res) {
            local_res = local_value;
            centerX = localX;
            centerY = localY;
          }
        }
        localX += step;
        world.barrier();
      }
      localY -= step;
    }
    boost::mpi::broadcast(world, centerX, 0);
    boost::mpi::broadcast(world, centerY, 0);
    boost::mpi::broadcast(world, local_res, 0);

    if (rank == 0) {
      // updating area
      step = 2 * step / static_cast<double>(size - 1);
      minX = centerX - step * (static_cast<double>(size - 1) / 2);
      maxY = centerY + step * (static_cast<double>(size - 1) / 2);
      info[0] = local_res;
      info[1] = step;
      info[2] = minX;
      info[3] = maxY;
    }

    // updating local result and information about area
    boost::mpi::broadcast(world, info, 0);

    if (rank != 0) {
      local_res = info[0];
      step = info[1];
      minX = info[2];
      maxY = info[3];
    }
  }

  if (rank == 0) {
    res = local_res;
  }
  return true;
}

bool ivanov_m_optimization_by_characteristics_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  }
  return true;
}