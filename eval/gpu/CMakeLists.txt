cmake_minimum_required(VERSION 3.18)
project(yuezu_fps_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90)

# 查找Thrust和CUB（通常随CUDA Toolkit一起提供）
find_package(CUDAToolkit REQUIRED)





# ==================== 优化版本 ====================
add_library(yuezu_fps_gpu_optimized STATIC
    yuezu_fps_gpu_optimized.cu
    yuezu_fps_gpu.h
)

target_include_directories(yuezu_fps_gpu_optimized PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(yuezu_fps_gpu_optimized PUBLIC
    CUDA::cudart
)

# 优化编译选项

set_target_properties(yuezu_fps_gpu_optimized PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)


# ==================== 优化编译选项 ====================
target_compile_options(yuezu_fps_gpu_optimized PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
    >
)


# ==================== 测试程序 ====================

# 优化版本测试
add_executable(test_gpu_fps_optimized
    test_gpu.cpp
)
target_link_libraries(test_gpu_fps_optimized PRIVATE
    yuezu_fps_gpu_optimized
)

# 优化版本对比测试
add_executable(compare_test_optimized
    compare_test.cpp
)
target_link_libraries(compare_test_optimized PRIVATE
    yuezu_fps_gpu_optimized
)

add_executable(compare_test_kitti
    compare_test_kitti.cpp
)
target_link_libraries(compare_test_kitti PRIVATE
    yuezu_fps_gpu_optimized
)

