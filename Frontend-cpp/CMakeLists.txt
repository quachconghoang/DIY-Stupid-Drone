cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_PREFIX_PATH "~/libtorch")
project(test_libtorch)

find_package( Torch REQUIRED )
find_package( CUDA REQUIRED )
find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

# file(GLOB ImgProcCUDA_src ./ImgProc/*.h ./ImgProc/*.cpp ./ImgProc/cuda/*.h ./ImgProc/cuda/*.cu)
# file(GLOB Utils_src ./Utils/*.hpp ./Utils/*.cpp)

add_executable(test_libtorch test_libtorch.cpp)
target_link_libraries(test_libtorch ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET test_libtorch PROPERTY CXX_STANDARD 11)