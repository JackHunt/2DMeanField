#Set CMake version and project name.
cmake_minimum_required(VERSION 2.8)
project(2DMeanField)

#Set compiler flags for Windows and Unix, respectively.
if(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
else(MSVC_IDE)
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -std=c++11 -fPIC -g -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -fPIC -O3 -march=native")
endif(MSVC_IDE)

find_package(OpenCV REQUIRED)

find_package(CUDA)
if(CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(WITH_CUDA TRUE)
  add_definitions(-DWITH_CUDA)
endif()

#Add shared library.
add_subdirectory(src/lib)

#Add demo.
add_subdirectory(src/app)
