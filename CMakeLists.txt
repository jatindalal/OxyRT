cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)

project(OxyRT)

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

add_executable(${PROJECT_NAME}
    src/main.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
    include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
    onnxruntime::onnxruntime
    ${OpenCV_LIBRARIES}
)

if(USE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)
endif()
if(USE_COREML)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_COREML)
endif()

add_executable(model_info
    src/model_info.cpp
)
target_link_libraries(model_info
    onnxruntime::onnxruntime
)

