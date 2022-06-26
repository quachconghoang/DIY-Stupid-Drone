cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_PREFIX_PATH "~/local")
project(ov_core)

find_package(Torch REQUIRED)
find_package(vpi REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package( OpenCV 4.5 QUIET )
if (NOT OpenCV_FOUND)
    find_package(OpenCV 4 REQUIRED)
endif ()
include_directories(${OpenCV_INCLUDE_DIRS})

find_package( Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB_RECURSE ELSED_LIB_SOURCES "ELSED/*.h")
list(APPEND ELSED_LIB_SOURCES
        ELSED/EdgeDrawer.cpp
        ELSED/ELSED.cpp
        ELSED/FullSegmentInfo.cpp)
add_library(elsed ${ELSED_LIB_SOURCES})
target_link_libraries(elsed ${OpenCV_LIBS})

file(GLOB_RECURSE DNN_LIB_SOURCES "DNN/*.h")
list(APPEND DNN_LIB_SOURCES
        DNN/Superpoint.cpp
        DNN/ANMS/anms.h
        DNN/ANMS/nanoflann.hpp
        DNN/ANMS/range-tree/lrtypes.h
        DNN/ANMS/range-tree/ranget.h
        )
add_library(mydnn ${DNN_LIB_SOURCES})
target_link_libraries(mydnn ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(graph_gen graph_gen.cpp)
target_link_libraries(graph_gen elsed mydnn "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
set_property(TARGET graph_gen PROPERTY CXX_STANDARD 14)

# add_executable(vpi_demo vpi_demo.cpp)
# target_link_libraries(vpi_demo vpi ${OpenCV_LIBS})