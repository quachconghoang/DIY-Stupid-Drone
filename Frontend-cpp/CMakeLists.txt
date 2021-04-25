cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_PREFIX_PATH "~/local")
project(test_libtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

# find_package( Eigen3 REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(test_libtorch test_libtorch.cpp)
target_link_libraries(test_libtorch "${TORCH_LIBRARIES}")
set_property(TARGET test_libtorch PROPERTY CXX_STANDARD 14)

#add_executable(test_superpoint Superpoint.cpp Superpoint.h test_Superpoint.cpp)
#target_link_libraries(test_superpoint ${TORCH_LIBRARIES} ${OpenCV_LIBS})
#set_property(TARGET test_superpoint PROPERTY CXX_STANDARD 14)
