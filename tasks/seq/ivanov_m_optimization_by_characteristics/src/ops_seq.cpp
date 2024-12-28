// Copyright 2024 Ivanov Mike
#include "seq/ivanov_m_optimization_by_characteristics/include/ops_seq.hpp"

bool ivanov_m_optimization_by_characteristics_seq::TestTaskSequential::pre_processing() {
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

bool ivanov_m_optimization_by_characteristics_seq::TestTaskSequential::validation() {
  internal_order_test();

  return (static_cast<int>(taskData->inputs.size()) == 1 && static_cast<int>(taskData->inputs_count.size()) == 1 &&
          static_cast<int>(taskData->inputs_count[0]) == 5 && static_cast<int>(taskData->outputs.size()) == 1 &&
          static_cast<int>(taskData->outputs_count.size()) == 1 && static_cast<int>(taskData->outputs_count[0]) == 1);
}

bool ivanov_m_optimization_by_characteristics_seq::TestTaskSequential::run() {
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

bool ivanov_m_optimization_by_characteristics_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
