cmake_minimum_required(VERSION 3.10)

# ── MNN OCR JNI ──────────────────────────────────────────
# 使用 prebuilt libMNN.so，不需要从源码编译 MNN

project("mnn_ocr")

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── jniLibs 路径 ─────────────────────────────────────────
# MNN 的 .so 最终会打包到 APK 的 jniLibs 下
# 这里通过 IMPORTED 库让 CMake 在链接时能找到它
set(JNILIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")

# Prebuilt MNN shared library
add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES
    IMPORTED_LOCATION "${JNILIB_PATH}/libMNN.so"
)

# MNN 头文件路径（从 GitHub 下载）
set(MNN_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/mnn_include/include")

# ── 构建 MNN JNI 共享库 ──────────────────────────────
add_library(mnn_ocr SHARED mnn_ocr_jni.cpp)

target_include_directories(mnn_ocr PRIVATE
    ${MNN_INCLUDE_PATH}
)

target_link_libraries(mnn_ocr
    MNN                              # prebuilt libMNN.so
    jnigraphics                       # Android Bitmap 访问
    log                               # android_log_print
)

# ── ncnn OCR JNI（保持不变）────────────────────────────
set(NCNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../ncnn-20260526-android)

if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(NCNN_ARCH ${NCNN_ROOT}/arm64-v8a)
elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
    set(NCNN_ARCH ${NCNN_ROOT}/armeabi-v7a)
elseif(ANDROID_ABI STREQUAL "x86")
    set(NCNN_ARCH ${NCNN_ROOT}/x86)
elseif(ANDROID_ABI STREQUAL "x86_64")
    set(NCNN_ARCH ${NCNN_ROOT}/x86_64)
else()
    message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
endif()

set(ncnn_DIR ${NCNN_ARCH}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(ncnn_ocr SHARED ncnn_ocr_jni.cpp)

target_link_libraries(ncnn_ocr
    ncnn
    jnigraphics
    log
)

target_include_directories(ncnn_ocr PRIVATE
    ${NCNN_ARCH}/include
)
