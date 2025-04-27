cmake_minimum_required(VERSION 3.10)
project(bag_of_visual_words)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(kmeans ../src/KMeans.cpp)

add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main kmeans)