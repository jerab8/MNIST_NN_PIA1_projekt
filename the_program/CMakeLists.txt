cmake_minimum_required(VERSION 3.10)
project(MNIST_NN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executables
add_executable(train train.cpp)
add_executable(predict predict.cpp)
