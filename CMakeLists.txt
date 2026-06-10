cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)

project(OxyRT)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
    src/main.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
    include/
    external/onnxruntime-osx-arm64-1.26.0/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(${PROJECT_NAME} PUBLIC
    external/onnxruntime-osx-arm64-1.26.0/lib
)
target_link_libraries(${PROJECT_NAME}
    onnxruntime
    ${OpenCV_LIBRARIES}
)

add_executable(model_info
    src/model_info.cpp
)
target_include_directories(model_info PUBLIC
    external/onnxruntime-osx-arm64-1.26.0/include
)
target_link_directories(model_info PUBLIC
    external/onnxruntime-osx-arm64-1.26.0/lib
)
target_link_libraries(model_info
    onnxruntime
)

