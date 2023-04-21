cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

project(long-tail)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(torchtrt REQUIRED)

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

add_library(long-tail ${SOURCES})
target_link_libraries(long-tail ${TORCH_LIBRARIES} TorchVision::TorchVision torchtrt nlohmann_json::nlohmann_json)

add_executable(example ${PROJECT_SOURCE_DIR}/example.cpp)
target_link_libraries(example long-tail)

add_executable(test_lowering ${PROJECT_SOURCE_DIR}/testing/test_lowering.cpp)
target_link_libraries(test_lowering long-tail)

add_executable(run_trt ${PROJECT_SOURCE_DIR}/run_trt.cpp)
target_link_libraries(run_trt long-tail)

add_executable(run_net_trt ${PROJECT_SOURCE_DIR}/run_net_trt.cpp)
target_link_libraries(run_net_trt long-tail)
