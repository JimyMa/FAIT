cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

project(long-tail)

cmake_policy(SET CMP0074 NEW)

# Options
option(ENABLE_TORCH_TRT "enable torch-trt" OFF)

# Packages
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
if(ENABLE_TORCH_TRT)
  find_package(torchtrt REQUIRED)
endif(ENABLE_TORCH_TRT)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(json)

include_directories(${PROJECT_SOURCE_DIR})

file(GLOB SOURCES
  ${PROJECT_SOURCE_DIR}/fuser/*.cpp
  ${PROJECT_SOURCE_DIR}/passes/*.cpp
  ${PROJECT_SOURCE_DIR}/tensorexpr/*.cpp
  ${PROJECT_SOURCE_DIR}/util/*.cpp)

set(LONG_TAIL_LINK_LIBS 
  ${TORCH_LIBRARIES}
  TorchVision::TorchVision
  nlohmann_json::nlohmann_json
  CUDA::cupti)

if(ENABLE_TORCH_TRT)
  set(LONG_TAIL_LINK_LIBS ${LONG_TAIL_LINK_LIBS} torchtrt)
endif(ENABLE_TORCH_TRT)

add_library(long-tail ${SOURCES})
target_link_libraries(long-tail ${LONG_TAIL_LINK_LIBS})

add_executable(example ${PROJECT_SOURCE_DIR}/example.cpp)
target_link_libraries(example long-tail)

add_executable(run_ts ${PROJECT_SOURCE_DIR}/run_ts.cpp)
target_link_libraries(run_ts long-tail)

add_executable(test_lowering ${PROJECT_SOURCE_DIR}/testing/test_lowering.cpp)
target_link_libraries(test_lowering long-tail)

if(ENABLE_TORCH_TRT)
  add_executable(run_trt ${PROJECT_SOURCE_DIR}/run_trt.cpp)
  target_link_libraries(run_trt long-tail)

  add_executable(run_net_trt ${PROJECT_SOURCE_DIR}/run_net_trt.cpp)
  target_link_libraries(run_net_trt long-tail)
endif(ENABLE_TORCH_TRT)
