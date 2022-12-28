cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

project(long-tail)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

include_directories(${PROJECT_SOURCE_DIR})

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/passes/*.cpp ${PROJECT_SOURCE_DIR}/util/*.cpp)
add_executable(example ${SOURCES})
target_link_libraries(example ${TORCH_LIBRARIES} TorchVision::TorchVision)
