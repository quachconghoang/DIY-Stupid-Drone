cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
# set(CMAKE_PREFIX_PATH "~/local")
project(gtsam-examples)

# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# find_package( OpenCV REQUIRED )
# include_directories(${OpenCV_INCLUDE_DIRS})

find_package( Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Find GTSAM components
find_package(GTSAM REQUIRED) # Uses installed package
set(GTSAM_LIBRARIES gtsam)   # TODO: automatic search libs
find_package(GTSAMCMakeTools)
include(GtsamMakeConfigFile)
# include(GtsamBuildTypes)
# include(GtsamTesting)
include(GtsamPrinting)
include(GtsamBuildTypes) # Load build type flags and default to Debug mode


#include(GtsamMatlabWrap) # Automatic MATLAB wrapper generation

include_directories(${GTSAM_INCLUDE_DIR})


add_executable(helloGTSAM hello_gtsam.cpp)
target_link_libraries(helloGTSAM gtsam)

#add_executable(test_superpoint Superpoint.cpp Superpoint.h test_Superpoint.cpp)
#target_link_libraries(test_superpoint ${TORCH_LIBRARIES} ${OpenCV_LIBS})
#set_property(TARGET test_superpoint PROPERTY CXX_STANDARD 14)