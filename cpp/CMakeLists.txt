cmake_minimum_required(VERSION 3.16)

project(ImageForgeryDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    features2d
    calib3d
    flann
)

add_executable(forgery_detector
    main.cpp
    Preprocessor.cpp
    DCTDetector.cpp
    FeatureMatcher.cpp
    GeometricVerifier.cpp
    MaskGenerator.cpp
)

target_include_directories(forgery_detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(forgery_detector PRIVATE
    ${OpenCV_LIBS}
)

if(MSVC)
    target_compile_options(forgery_detector PRIVATE /W4)
else()
    target_compile_options(forgery_detector PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()