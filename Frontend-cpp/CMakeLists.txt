cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_PREFIX_PATH "~/local")
project(SuperPoint_For_TX2)

find_package( Torch REQUIRED )
find_package( CUDA REQUIRED )
find_package( OpenCV REQUIRED )
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${OpenCV_INCLUDE_DIRS})

# file(GLOB ImgProcCUDA_src ./ImgProc/*.h ./ImgProc/*.cpp ./ImgProc/cuda/*.h ./ImgProc/cuda/*.cu)
# file(GLOB Utils_src ./Utils/*.hpp ./Utils/*.cpp)

#add_executable(test_libtorch test_libtorch.cpp)
#target_link_libraries(test_libtorch ${TORCH_LIBRARIES} ${OpenCV_LIBS})
#set_property(TARGET test_libtorch PROPERTY CXX_STANDARD 11)

add_executable(test_superpoint Superpoint.cpp Superpoint.h test_Superpoint.cpp)
target_link_libraries(test_superpoint ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET test_superpoint PROPERTY CXX_STANDARD 11)