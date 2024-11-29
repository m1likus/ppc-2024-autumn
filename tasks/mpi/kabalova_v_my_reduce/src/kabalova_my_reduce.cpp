// Copyright 2024 Kabalova Valeria
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "mpi/kabalova_v_my_reduce/include/kabalova_my_reduce.hpp"

template <typename T>
bool kabalova_v_my_reduce::TestMPITaskParallel<T>::pre_processing() {
  internal_order_test();
  
  return true;
}

template <typename T>
bool kabalova_v_my_reduce::TestMPITaskParallel<T>::validation() {
  internal_order_test();
  
  return true;
}

template <typename T>
bool kabalova_v_my_reduce::TestMPITaskParallel<T>::run() {
  internal_order_test();
  
  return true;
}

template <typename T>
bool kabalova_v_my_reduce::TestMPITaskParallel<T>::post_processing() {
  internal_order_test();
  
  return true;
}