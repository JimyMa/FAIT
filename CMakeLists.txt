cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

project(long-tail)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

include_directories(${PROJECT_SOURCE_DIR})

file(GLOB SOURCES
        ${PROJECT_SOURCE_DIR}/fuser/*.cpp
        ${PROJECT_SOURCE_DIR}/passes/*.cpp
        ${PROJECT_SOURCE_DIR}/tensorexpr/*.cpp
        ${PROJECT_SOURCE_DIR}/util/*.cpp)
file(GLOB EXAMPLE ${PROJECT_SOURCE_DIR}/example.cpp)
file(GLOB NNC_EXAMPLE ${PROJECT_SOURCE_DIR}/nnc_example.cpp)
file(GLOB NNC_EXAMPLE_DYN_SHAPE ${PROJECT_SOURCE_DIR}/nnc_example_dyn_shape.cpp)
file(GLOB SIMPLE_LOOP_NNC ${PROJECT_SOURCE_DIR}/simple_loop_nnc.cpp)


message(${TORCH_LIBRARIES})

add_library(long-tail ${SOURCES})
target_link_libraries(long-tail ${TORCH_LIBRARIES} TorchVision::TorchVision)

add_executable(example ${EXAMPLE})
target_link_libraries(example long-tail)

# add_executable(nnc_example ${NNC_EXAMPLE})
# target_link_libraries(nnc_example long-tail)

# add_executable(nnc_example_dyn_shape ${NNC_EXAMPLE_DYN_SHAPE})
# target_link_libraries(nnc_example_dyn_shape long-tail)

# add_executable(simple_loop_nnc ${SIMPLE_LOOP_NNC})
# target_link_libraries(simple_loop_nnc long-tail)






