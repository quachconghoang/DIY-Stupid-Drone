cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_PREFIX_PATH "~/local")
project(test_SP)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

find_package( CUDA REQUIRED )
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
set(CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75;-rdc=true;-use_fast_math")
include_directories(${CUDA_INCLUDE_DIRS})

# find_package( Eigen3 REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(test_SP test_Superpoint.cpp SuperPoint.h SuperPoint.cpp)
target_link_libraries(test_SP "${TORCH_LIBRARIES}")
set_property(TARGET test_SP PROPERTY CXX_STANDARD 14)

target_link_libraries(test_SP ${OpenCV_LIBS} ${CUDA_LIBRARIES})

#add_executable(test_superpoint Superpoint.cpp Superpoint.h test_Superpoint.cpp)
#target_link_libraries(test_superpoint ${TORCH_LIBRARIES} ${OpenCV_LIBS})
#set_property(TARGET test_superpoint PROPERTY CXX_STANDARD 14)
