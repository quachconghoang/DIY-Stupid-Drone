cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
# set(CMAKE_PREFIX_PATH "~/local")
project(SuperPoint_For_Jetson)

# find_package( Torch REQUIRED )

find_package( CUDA REQUIRED )
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
# set(CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75;-rdc=true;-use_fast_math")
include_directories(${CUDA_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

# find_package( Eigen3 REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIR})

# Enable TensorRT CXX
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
message("${TensorRT_LIBRARY}")

# Enable TensorRT Plugins: FUCK!
list(APPEND PLUGINS "nvinfer")
list(APPEND PLUGINS "nvonnxparser")
list(APPEND PLUGINS "nvparsers")
list(APPEND PLUGINS "nvinfer_plugin")
foreach(libName ${PLUGINS})
    find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
    list(APPEND PLUGIN_LIBS "${${libName}_lib}")
endforeach()
message("${PLUGIN_LIBS}")

#add_executable(test_libtorch test_libtorch.cpp)
#target_link_libraries(test_libtorch ${TORCH_LIBRARIES} ${OpenCV_LIBS})
#set_property(TARGET test_libtorch PROPERTY CXX_STANDARD 11)

#add_executable(test_superpoint Superpoint.cpp Superpoint.h test_Superpoint.cpp)
#target_link_libraries(test_superpoint ${TORCH_LIBRARIES} ${OpenCV_LIBS})
#set_property(TARGET test_superpoint PROPERTY CXX_STANDARD 14)

# Add Executable
add_executable(testTRT test_TRT.cpp)

target_link_libraries(testTRT ${OpenCV_LIBS} ${TensorRT_LIBRARY} ${CUDA_LIBRARIES})