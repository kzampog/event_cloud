cmake_minimum_required(VERSION 2.8)
project(test_project)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# compile with C++11 support
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-O3 -march=native")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3 -march=native")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Pangolin REQUIRED)
find_package(cilantro REQUIRED)

set(INCLUDES ${Pangolin_INCLUDE_DIRS} ${cilantro_INCLUDE_DIRS})

set(LIBRARIES ${Pangolin_LIBRARIES} ${cilantro_LIBRARIES})

include_directories(include)
include_directories(${INCLUDES})

# Targets

add_executable(dvs_test src/dvs_test.cpp)
target_link_libraries(dvs_test ${LIBRARIES})

add_executable(dvs_warp_field_2d src/dvs_warp_field_2d.cpp)
target_link_libraries(dvs_warp_field_2d ${LIBRARIES})

add_executable(dvs_warp_field_hybrid src/dvs_warp_field_hybrid.cpp)
target_link_libraries(dvs_warp_field_hybrid ${LIBRARIES})

add_executable(dvs_warp_field_3d src/dvs_warp_field_3d.cpp)
target_link_libraries(dvs_warp_field_3d ${LIBRARIES})
