# References
# https://github.com/bast/hdf5-cmake-example/blob/master/CMakeLists.txt
# http://docs.opencv.org/3.1.0/db/df5/tutorial_linux_gcc_cmake.html
# https://cmake.org/Wiki/CMake:How_To_Find_Libraries
# http://stackoverflow.com/questions/22598208/linking-opencv-libraries-included-as-an-external-project-via-cmake
# http://derekmolloy.ie/hello-world-introductions-to-cmake/
# http://stackoverflow.com/questions/21163188/most-simple-but-complete-cmake-example
# http://blog.beuc.net/posts/Cross-compiling_with_CMake/

# Running with Debug
# cmake .. -DCMAKE_BUILD_TYPE=Debug; make -j8
cmake_minimum_required (VERSION 3.1)

# Set C++11
enable_language(CXX)
set (CMAKE_CXX_STANDARD 11)

find_package(HDF5)
if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIR})
    set(_hdf5_libs hdf5 hdf5_cpp hdf5_hl_cpp)
endif()

# get all *.cpp and header files recursively (Not a good idea for cpp files)
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HDRS *.h)

# Remove CMakeCXXCompilerId.cpp from cpp list (Otherwise we get multiple main error)
file(GLOB_RECURSE REMOVE_CMAKE "CMakeCXXCompilerId.cpp")
list(REMOVE_ITEM SRC_FILES ${REMOVE_CMAKE})

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")

add_executable(Skeleton ${SRC_FILES})
target_link_libraries(Skeleton ${_hdf5_libs})
