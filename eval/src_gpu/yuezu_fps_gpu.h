#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// =============================================================================
// 配置宏定义 - 可调参数
// =============================================================================

// 每个MortonBlock存储的点数
#ifndef MORTON_BLOCK_SIZE
#define MORTON_BLOCK_SIZE 16
#endif

// Chunk大小（对应GPU Thread Block大小）
#ifndef CHUNK_SIZE
#define CHUNK_SIZE 1024
#endif

// 默认各维度分块数
#ifndef DEFAULT_X_BLOCKS
#define DEFAULT_X_BLOCKS 16
#endif

#ifndef DEFAULT_Y_BLOCKS
#define DEFAULT_Y_BLOCKS 16
#endif

#ifndef DEFAULT_Z_BLOCKS
#define DEFAULT_Z_BLOCKS 16
#endif

// 边界 epsilon
#ifndef BOUNDARY_EPS
#define BOUNDARY_EPS 1e-6f
#endif

// 初始距离值
#ifndef INF_DISTANCE
#define INF_DISTANCE 1e30f
#endif

// =============================================================================
// 主机端数据结构
// =============================================================================

struct Point3D {
  float x, y, z;
  size_t original_index;

  Point3D(float x_ = 0, float y_ = 0, float z_ = 0, size_t idx = 0)
      : x(x_), y(y_), z(z_), original_index(idx) {}
};


struct SpaceRange {
  // 原有成员：包围盒与块数
  float min_x, max_x;
  float min_y, max_y;
  float min_z, max_z;
  uint32_t x_blocks, y_blocks, z_blocks;

  // 【新增】预计算的逆块大小 (避免除法)
  float inv_bs_x, inv_bs_y, inv_bs_z;

  // 【新增】每个维度占用的位数
  uint32_t x_bits_, y_bits_, z_bits_;

};

SpaceRange make_uniform_range(float min_x, float max_x, float min_y,
                              float max_y, float min_z, float max_z,
                              uint32_t blocks_per_dim);

SpaceRange make_range(float min_x, float max_x, float min_y, float max_y,
                      float min_z, float max_z, uint32_t x_blocks,
                      uint32_t y_blocks, uint32_t z_blocks);

SpaceRange compute_bbox(const std::vector<Point3D>& points,
                        uint32_t blocks_per_dim = DEFAULT_X_BLOCKS);

SpaceRange compute_bbox(const std::vector<Point3D>& points, uint32_t x_blocks,
                        uint32_t y_blocks, uint32_t z_blocks);

// =============================================================================
// GPU FPS 类
// =============================================================================

class FpsGPU {
public:
  FpsGPU();
  ~FpsGPU();

  // 新增：莫顿码 LUT
  std::vector<uint32_t> h_x_lut_; // Host 端临时存储
  std::vector<uint32_t> h_y_lut_;
  std::vector<uint32_t> h_z_lut_;
  uint32_t* d_x_lut_; // Device 端指针
  uint32_t* d_y_lut_;
  uint32_t* d_z_lut_;

  // 初始化：分配显存，预处理点云
  bool initialize(const std::vector<Point3D>& points, const SpaceRange& range);

  // 执行FPS采样
  std::vector<size_t> sample(size_t sample_count);

  // 释放资源
  void release();

private:
  // 禁用拷贝
  FpsGPU(const FpsGPU&) = delete;
  FpsGPU& operator=(const FpsGPU&) = delete;

  // 内部数据
  void* impl_;  // 指向内部实现的不透明指针
};

// =============================================================================
// 主入口函数
// =============================================================================

std::vector<size_t> yuezu_fps_gpu(const std::vector<Point3D>& point_cloud,
                                   size_t sample_count, const SpaceRange& range);
