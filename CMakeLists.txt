cmake_minimum_required(VERSION 3.5)

project(CID)

set(CMAKE_PREFIX_PATH "C:/Users/lopdl/Downloads/libtorch")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} /std:c++20")

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)