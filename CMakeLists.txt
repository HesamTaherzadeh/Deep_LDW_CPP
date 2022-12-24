cmake_minimum_required(VERSION 3.0.0)
project(ldw VERSION 0.1.0)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(TF_LITE_DIRECTORY "/home/hesam/tensorflow-2.6.0/tensorflow/lite")


# including two directory containing the header files 
INCLUDE_DIRECTORIES(~/tensorflow-2.6.0)
INCLUDE_DIRECTORIES("${TF_LITE_DIRECTORY}/tools/make/downloads/flatbuffers/include")

include_directories(${OpenCV_INCLUDE_DIRS})

#setting version of Cpp to be used 
set(CMAKE_CXX_STANDARD 17)
add_executable(${PROJECT_NAME} src/main.cpp src/video_opener.cpp src/inference.cpp)

# finding eigen and threads and opencv package
find_package (Eigen3 3.3 REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

#targeting libraries
target_link_libraries(${PROJECT_NAME} Threads::Threads)
target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${CMAKE_DL_LIBS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )
target_link_libraries(${PROJECT_NAME} "${TF_LITE_DIRECTORY}/tools/make/downloads/flatbuffers/build/libflatbuffers.a")
target_link_libraries(${PROJECT_NAME} "${TF_LITE_DIRECTORY}/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a")
# target_include_directories(${PROJECT_NAME} PUBLIC ${inference})


include(CPack)

