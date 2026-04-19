#include "yuezu_fps_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

// =============================================================================
// 工具函数实现
// =============================================================================

SpaceRange make_uniform_range(float min_x, float max_x, float min_y,
                              float max_y, float min_z, float max_z,
                              uint32_t blocks_per_dim) {
  return {min_x, max_x, min_y, max_y, min_z, max_z,
          blocks_per_dim, blocks_per_dim, blocks_per_dim};
}

SpaceRange make_range(float min_x, float max_x, float min_y, float max_y,
                      float min_z, float max_z, uint32_t x_blocks,
                      uint32_t y_blocks, uint32_t z_blocks) {
  return {min_x, max_x, min_y, max_y, min_z, max_z,
          x_blocks, y_blocks, z_blocks};
}

SpaceRange compute_bbox(const std::vector<Point3D>& points,
                        uint32_t blocks_per_dim) {
  return compute_bbox(points, blocks_per_dim, blocks_per_dim, blocks_per_dim);
}

SpaceRange compute_bbox(const std::vector<Point3D>& points, uint32_t x_blocks,
                        uint32_t y_blocks, uint32_t z_blocks) {
  if (points.empty()) {
    return make_range(0, 1, 0, 1, 0, 1, 1, 1, 1);
  }

  float min_x = points[0].x, max_x = points[0].x;
  float min_y = points[0].y, max_y = points[0].y;
  float min_z = points[0].z, max_z = points[0].z;

  for (const auto& p : points) {
    min_x = std::min(min_x, p.x);
    max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y);
    max_y = std::max(max_y, p.y);
    min_z = std::min(min_z, p.z);
    max_z = std::max(max_z, p.z);
  }

  float eps = BOUNDARY_EPS;
  return make_range(min_x - eps, max_x + eps, min_y - eps, max_y + eps,
                    min_z - eps, max_z + eps, x_blocks, y_blocks, z_blocks);
}

// =============================================================================
// GPU设备端数据结构
// =============================================================================

// 点云数据结构（SoA布局）
struct PointCloudGPU {
  float* x;
  float* y;
  float* z;
  uint32_t* original_index;
  float* min_dist;
  uint32_t num_points;
};

// MortonBlock（GPU版本）
struct MortonBlockGPU {
  uint32_t point_idx[MORTON_BLOCK_SIZE];
  float block_max_dist;
  uint32_t max_point_idx;
  uint8_t count;
};

// Chunk元数据
struct ChunkAABB {
  float min_x, max_x;
  float min_y, max_y;
  float min_z, max_z;
};

// =============================================================================
// FpsGPU内部实现类
// =============================================================================

class FpsGPUImpl {
public:
  FpsGPUImpl()
      : num_points_(0), d_x(nullptr), d_y(nullptr), d_z(nullptr),
        d_original_index(nullptr), d_min_dist(nullptr), d_morton_code(nullptr),
        d_sorted_index(nullptr), d_blocks(nullptr), d_block_offsets(nullptr),
        d_chunk_aabb(nullptr), d_active_chunks(nullptr),
        d_block_max_vals(nullptr), d_block_max_indices(nullptr),
        d_argmax_result(nullptr), h_sampled_indices(nullptr) {}

  ~FpsGPUImpl() {
    release();
  }

  bool initialize(const std::vector<Point3D>& points, const SpaceRange& range);
  std::vector<size_t> sample(size_t sample_count);
  void release();

private:
  uint32_t num_points_;
  SpaceRange range_;

  // 设备端指针
  float* d_x;
  float* d_y;
  float* d_z;
  uint32_t* d_original_index;
  float* d_min_dist;
  uint32_t* d_morton_code;
  uint32_t* d_sorted_index;
  MortonBlockGPU* d_blocks;
  uint32_t* d_block_offsets;
  ChunkAABB* d_chunk_aabb;
  int32_t* d_active_chunks;
  float* d_block_max_vals;
  uint32_t* d_block_max_indices;
  uint32_t* d_argmax_result;

  // 主机端数据
  uint32_t* h_sampled_indices;

  // 块信息
  uint32_t num_blocks_;
  uint32_t num_chunks_;
};

// =============================================================================
// GPU内核：计算Morton编码
// =============================================================================

__device__ __forceinline__ uint32_t compute_morton_bits(uint32_t x) {
  x = (x | (x << 16)) & 0x030000FF;
  x = (x | (x << 8))  & 0x0300F00F;
  x = (x | (x << 4))  & 0x030C30C3;
  x = (x | (x << 2))  & 0x09249249;
  return x;
}

__device__ __forceinline__ uint32_t encode_morton(uint32_t x, uint32_t y, uint32_t z) {
  return compute_morton_bits(x) |
         (compute_morton_bits(y) << 1) |
         (compute_morton_bits(z) << 2);
}

__global__ void compute_morton_codes_kernel(
    const float* x, const float* y, const float* z,
    uint32_t* morton_codes, uint32_t* indices,
    float min_x, float min_y, float min_z,
    float inv_block_size_x, float inv_block_size_y, float inv_block_size_z,
    uint32_t x_blocks, uint32_t y_blocks, uint32_t z_blocks,
    uint32_t num_points) {

  uint32_t idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= num_points) return;

  float px = x[idx];
  float py = y[idx];
  float pz = z[idx];

  // 计算块坐标
  uint32_t bx = static_cast<uint32_t>((px - min_x) * inv_block_size_x);
  uint32_t by = static_cast<uint32_t>((py - min_y) * inv_block_size_y);
  uint32_t bz = static_cast<uint32_t>((pz - min_z) * inv_block_size_z);

  bx = min(bx, x_blocks - 1);
  by = min(by, y_blocks - 1);
  bz = min(bz, z_blocks - 1);

  morton_codes[idx] = encode_morton(bx, by, bz);
  indices[idx] = idx;
}

// =============================================================================
// GPU内核：初始化距离数组
// =============================================================================

__global__ void init_min_dist_kernel(float* min_dist, uint32_t num_points) {
  uint32_t idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= num_points) return;
  min_dist[idx] = INF_DISTANCE;
}

// =============================================================================
// GPU内核：更新第一个采样点的距离
// =============================================================================

__global__ void update_first_point_kernel(
    const float* x, const float* y, const float* z,
    float* min_dist, float qx, float qy, float qz, uint32_t num_points) {

  uint32_t idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= num_points) return;

  float dx = x[idx] - qx;
  float dy = y[idx] - qy;
  float dz = z[idx] - qz;

  min_dist[idx] = dx * dx + dy * dy + dz * dz;
}

// =============================================================================
// GPU内核：构建MortonBlock
// =============================================================================

__global__ void build_morton_blocks_kernel(
    const uint32_t* sorted_indices,
    const float* x, const float* y, const float* z,
    MortonBlockGPU* blocks, uint32_t num_points) {

  uint32_t block_idx = blockIdx.x;
  uint32_t start = block_idx * MORTON_BLOCK_SIZE;
  uint32_t end = min(start + MORTON_BLOCK_SIZE, num_points);

  if (start >= num_points) return;

  MortonBlockGPU blk;
  blk.count = static_cast<uint8_t>(end - start);
  blk.block_max_dist = 0.0f;
  blk.max_point_idx = 0;

  for (uint32_t i = 0; i < blk.count; ++i) {
    blk.point_idx[i] = sorted_indices[start + i];
  }

  blocks[block_idx] = blk;
}

// =============================================================================
// GPU内核：构建Chunk AABB
// =============================================================================

__global__ void build_chunk_aabb_kernel(
    const float* x, const float* y, const float* z,
    const uint32_t* sorted_indices,
    ChunkAABB* aabbs, uint32_t num_points) {

  uint32_t chunk_idx = blockIdx.x;
  uint32_t start = chunk_idx * CHUNK_SIZE;
  uint32_t end = min(start + CHUNK_SIZE, num_points);

  if (start >= num_points) return;

  float min_x = INF_DISTANCE, max_x = -INF_DISTANCE;
  float min_y = INF_DISTANCE, max_y = -INF_DISTANCE;
  float min_z = INF_DISTANCE, max_z = -INF_DISTANCE;

  for (uint32_t i = start; i < end; ++i) {
    uint32_t idx = sorted_indices[i];
    float px = x[idx];
    float py = y[idx];
    float pz = z[idx];

    min_x = min(min_x, px);
    max_x = max(max_x, px);
    min_y = min(min_y, py);
    max_y = max(max_y, py);
    min_z = min(min_z, pz);
    max_z = max(max_z, pz);
  }

  aabbs[chunk_idx].min_x = min_x;
  aabbs[chunk_idx].max_x = max_x;
  aabbs[chunk_idx].min_y = min_y;
  aabbs[chunk_idx].max_y = max_y;
  aabbs[chunk_idx].min_z = min_z;
  aabbs[chunk_idx].max_z = max_z;
}

// =============================================================================
// GPU内核：筛选活跃Chunk
// =============================================================================

__device__ __forceinline__ float point_to_aabb_dist_sq(
    float px, float py, float pz,
    float min_x, float max_x,
    float min_y, float max_y,
    float min_z, float max_z) {

  float dx = max(max(min_x - px, 0.0f), px - max_x);
  float dy = max(max(min_y - py, 0.0f), py - max_y);
  float dz = max(max(min_z - pz, 0.0f), pz - max_z);

  return dx * dx + dy * dy + dz * dz;
}

__global__ void filter_active_chunks_kernel(
    const ChunkAABB* aabbs, uint32_t num_chunks,
    float qx, float qy, float qz, float max_dist_sq,
    int32_t* active_chunks) {

  uint32_t idx = blockIdx.x;
  if (idx >= num_chunks) return;

  const ChunkAABB& aabb = aabbs[idx];
  float dist_sq = point_to_aabb_dist_sq(qx, qy, qz,
                                          aabb.min_x, aabb.max_x,
                                          aabb.min_y, aabb.max_y,
                                          aabb.min_z, aabb.max_z);

  active_chunks[idx] = (dist_sq <= max_dist_sq) ? 1 : 0;
}

// =============================================================================
// GPU内核：更新活跃Chunk内的点
// =============================================================================

__global__ void update_active_chunks_kernel(
    const float* x, const float* y, const float* z,
    const uint32_t* sorted_indices,
    const int32_t* active_chunks,
    float* min_dist,
    float qx, float qy, float qz,
    uint32_t num_points, uint32_t num_chunks) {

  uint32_t chunk_idx = blockIdx.x;
  if (chunk_idx >= num_chunks || active_chunks[chunk_idx] == 0) return;

  uint32_t start = chunk_idx * CHUNK_SIZE;
  uint32_t end = min(start + CHUNK_SIZE, num_points);

  for (uint32_t i = start + threadIdx.x; i < end; i += 64) {
    uint32_t idx = sorted_indices[i];
    float dx = x[idx] - qx;
    float dy = y[idx] - qy;
    float dz = z[idx] - qz;
    float dist_sq = dx * dx + dy * dy + dz * dz;

    if (dist_sq < min_dist[idx]) {
      min_dist[idx] = dist_sq;
    }
  }
}

// =============================================================================
// GPU内核：分块ArgMax归约
// =============================================================================

__global__ void block_argmax_kernel(const float* data, uint32_t num_points,
                                      float* block_max_vals, uint32_t* block_max_indices) {
  const int block_size = 1024;
  __shared__ float sh_max[1024];
  __shared__ uint32_t sh_idx[1024];

  int tid = threadIdx.x;
  int block_id = blockIdx.x;

  // 初始化共享内存
  if (tid < 1024) {
    sh_max[tid] = -INF_DISTANCE;
    sh_idx[tid] = 0;
  }
  __syncthreads();

  float max_val = -INF_DISTANCE;
  uint32_t max_idx = 0;

  // 第一步：每个线程读取多个元素
  for (uint32_t i = block_id * block_size + tid; i < num_points; i += gridDim.x * block_size) {
    float val = data[i];
    if (val > max_val) {
      max_val = val;
      max_idx = i;
    }
  }

  // 第二步：写入共享内存
  if (tid < 1024) {
    sh_max[tid] = max_val;
    sh_idx[tid] = max_idx;
  }
  __syncthreads();

  // 第三步：块内归约
  for (int s = 512; s > 0; s >>= 1) {
    if (tid < s) {
      if (sh_max[tid] < sh_max[tid + s]) {
        sh_max[tid] = sh_max[tid + s];
        sh_idx[tid] = sh_idx[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_max_vals[block_id] = sh_max[0];
    block_max_indices[block_id] = sh_idx[0];
  }
}

__global__ void final_argmax_kernel(const float* block_vals, const uint32_t* block_indices,
                                     uint32_t num_blocks, uint32_t* result) {
  const int block_size = 1024;
  __shared__ float sh_max[1024];
  __shared__ uint32_t sh_idx[1024];

  int tid = threadIdx.x;

  // 初始化共享内存
  if (tid < 1024) {
    sh_max[tid] = -INF_DISTANCE;
    sh_idx[tid] = 0;
  }
  __syncthreads();

  float max_val = -INF_DISTANCE;
  uint32_t max_idx = 0;

  // 第一步：每个线程读取多个元素
  for (uint32_t i = tid; i < num_blocks; i += block_size) {
    float val = block_vals[i];
    if (val > max_val) {
      max_val = val;
      max_idx = block_indices[i];
    }
  }

  // 第二步：写入共享内存
  if (tid < 1024) {
    sh_max[tid] = max_val;
    sh_idx[tid] = max_idx;
  }
  __syncthreads();

  // 第三步：块内归约
  for (int s = 512; s > 0; s >>= 1) {
    if (tid < s) {
      if (sh_max[tid] < sh_max[tid + s]) {
        sh_max[tid] = sh_max[tid + s];
        sh_idx[tid] = sh_idx[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *result = sh_idx[0];
  }
}

// =============================================================================
// FpsGPUImpl实现
// =============================================================================

bool FpsGPUImpl::initialize(const std::vector<Point3D>& points, const SpaceRange& range) {
  num_points_ = static_cast<uint32_t>(points.size());
  range_ = range;

  if (num_points_ == 0) return false;

  // 分配设备内存
  cudaMalloc(&d_x, num_points_ * sizeof(float));
  cudaMalloc(&d_y, num_points_ * sizeof(float));
  cudaMalloc(&d_z, num_points_ * sizeof(float));
  cudaMalloc(&d_original_index, num_points_ * sizeof(uint32_t));
  cudaMalloc(&d_min_dist, num_points_ * sizeof(float));
  cudaMalloc(&d_morton_code, num_points_ * sizeof(uint32_t));
  cudaMalloc(&d_sorted_index, num_points_ * sizeof(uint32_t));

  // 拷贝点云数据到设备
  std::vector<float> h_x(num_points_), h_y(num_points_), h_z(num_points_);
  std::vector<uint32_t> h_original_index(num_points_);

  for (uint32_t i = 0; i < num_points_; ++i) {
    h_x[i] = points[i].x;
    h_y[i] = points[i].y;
    h_z[i] = points[i].z;
    h_original_index[i] = static_cast<uint32_t>(points[i].original_index);
  }

  cudaMemcpy(d_x, h_x.data(), num_points_ * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y.data(), num_points_ * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, h_z.data(), num_points_ * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_original_index, h_original_index.data(), num_points_ * sizeof(uint32_t), cudaMemcpyHostToDevice);

  // 计算Morton编码
  float inv_block_size_x = 1.0f / ((range.max_x - range.min_x) / range.x_blocks);
  float inv_block_size_y = 1.0f / ((range.max_y - range.min_y) / range.y_blocks);
  float inv_block_size_z = 1.0f / ((range.max_z - range.min_z) / range.z_blocks);

  uint32_t blocks = (num_points_ + 255) / 256;
  compute_morton_codes_kernel<<<blocks, 256>>>(
      d_x, d_y, d_z, d_morton_code, d_sorted_index,
      range.min_x, range.min_y, range.min_z,
      inv_block_size_x, inv_block_size_y, inv_block_size_z,
      range.x_blocks, range.y_blocks, range.z_blocks, num_points_);

  cudaDeviceSynchronize();

  // 使用Thrust按Morton码排序
  thrust::device_ptr<uint32_t> dev_morton(d_morton_code);
  thrust::device_ptr<uint32_t> dev_index(d_sorted_index);
  thrust::sort_by_key(dev_morton, dev_morton + num_points_, dev_index);

  cudaDeviceSynchronize();

  // 构建MortonBlock
  num_blocks_ = (num_points_ + MORTON_BLOCK_SIZE - 1) / MORTON_BLOCK_SIZE;
  cudaMalloc(&d_blocks, num_blocks_ * sizeof(MortonBlockGPU));
  build_morton_blocks_kernel<<<num_blocks_, 1>>>(
      d_sorted_index, d_x, d_y, d_z, d_blocks, num_points_);

  cudaDeviceSynchronize();

  // 构建Chunk AABB
  num_chunks_ = (num_points_ + CHUNK_SIZE - 1) / CHUNK_SIZE;
  cudaMalloc(&d_chunk_aabb, num_chunks_ * sizeof(ChunkAABB));
  build_chunk_aabb_kernel<<<num_chunks_, 1>>>(
      d_x, d_y, d_z, d_sorted_index, d_chunk_aabb, num_points_);

  cudaDeviceSynchronize();

  // 分配活跃Chunk标记数组
  cudaMalloc(&d_active_chunks, num_chunks_ * sizeof(int32_t));

  // 为两级ArgMax分配临时存储（固定64个块）
  const uint32_t argmax_num_blocks = 64;
  cudaMalloc(&d_block_max_vals, argmax_num_blocks * sizeof(float));
  cudaMalloc(&d_block_max_indices, argmax_num_blocks * sizeof(uint32_t));
  cudaMalloc(&d_argmax_result, sizeof(uint32_t));

  // 分配主机端采样索引数组
  h_sampled_indices = new uint32_t[num_points_];

  return true;
}

std::vector<size_t> FpsGPUImpl::sample(size_t sample_count) {
  if (num_points_ == 0 || sample_count == 0) return {};

  sample_count = std::min(sample_count, static_cast<size_t>(num_points_));

  std::vector<size_t> result;
  result.reserve(sample_count);

  // 初始化距离数组为无穷大
  uint32_t blocks = (num_points_ + 255) / 256;
  init_min_dist_kernel<<<blocks, 256>>>(d_min_dist, num_points_);
  cudaDeviceSynchronize();

  // 第一个采样点：固定为index 0
  uint32_t first_idx = 0;

  float h_qx, h_qy, h_qz;
  cudaMemcpy(&h_qx, d_x + first_idx, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_qy, d_y + first_idx, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_qz, d_z + first_idx, sizeof(float), cudaMemcpyDeviceToHost);

  uint32_t h_orig_idx;
  cudaMemcpy(&h_orig_idx, d_original_index + first_idx, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  result.push_back(h_orig_idx);
  h_sampled_indices[0] = first_idx;

  // 更新第一个点的距离
  update_first_point_kernel<<<blocks, 256>>>(
      d_x, d_y, d_z, d_min_dist, h_qx, h_qy, h_qz, num_points_);
  cudaDeviceSynchronize();

  float current_max_dist_sq = INF_DISTANCE;

  // 迭代采样
  for (size_t i = 1; i < sample_count; ++i) {
    // 使用两级argmax找到最远点
    const uint32_t argmax_num_blocks = 64;
    block_argmax_kernel<<<argmax_num_blocks, 1024>>>(
        d_min_dist, num_points_, d_block_max_vals, d_block_max_indices);
    cudaDeviceSynchronize();

    final_argmax_kernel<<<1, 1024>>>(
        d_block_max_vals, d_block_max_indices, argmax_num_blocks, d_argmax_result);
    cudaDeviceSynchronize();

    uint32_t farthest_idx;
    cudaMemcpy(&farthest_idx, d_argmax_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 获取最远点的原始索引
    cudaMemcpy(&h_orig_idx, d_original_index + farthest_idx, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    result.push_back(h_orig_idx);
    h_sampled_indices[i] = farthest_idx;

    // 获取新采样点坐标
    cudaMemcpy(&h_qx, d_x + farthest_idx, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_qy, d_y + farthest_idx, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_qz, d_z + farthest_idx, sizeof(float), cudaMemcpyDeviceToHost);

    // 统一使用局部更新策略
    // 先获取当前最大距离
    float h_min_dist;
    cudaMemcpy(&h_min_dist, d_min_dist + farthest_idx, sizeof(float), cudaMemcpyDeviceToHost);
    current_max_dist_sq = h_min_dist;

    // 筛选活跃Chunk
    filter_active_chunks_kernel<<<num_chunks_, 1>>>(
        d_chunk_aabb, num_chunks_, h_qx, h_qy, h_qz, current_max_dist_sq, d_active_chunks);
    cudaDeviceSynchronize();

    // 更新活跃Chunk内的点
    update_active_chunks_kernel<<<num_chunks_, 64>>>(
        d_x, d_y, d_z, d_sorted_index, d_active_chunks,
        d_min_dist, h_qx, h_qy, h_qz, num_points_, num_chunks_);

    cudaDeviceSynchronize();
  }

  return result;
}

void FpsGPUImpl::release() {
  if (d_x) cudaFree(d_x);
  if (d_y) cudaFree(d_y);
  if (d_z) cudaFree(d_z);
  if (d_original_index) cudaFree(d_original_index);
  if (d_min_dist) cudaFree(d_min_dist);
  if (d_morton_code) cudaFree(d_morton_code);
  if (d_sorted_index) cudaFree(d_sorted_index);
  if (d_blocks) cudaFree(d_blocks);
  if (d_chunk_aabb) cudaFree(d_chunk_aabb);
  if (d_active_chunks) cudaFree(d_active_chunks);
  if (d_block_max_vals) cudaFree(d_block_max_vals);
  if (d_block_max_indices) cudaFree(d_block_max_indices);
  if (d_argmax_result) cudaFree(d_argmax_result);
  if (h_sampled_indices) delete[] h_sampled_indices;

  d_x = nullptr;
  d_y = nullptr;
  d_z = nullptr;
  d_original_index = nullptr;
  d_min_dist = nullptr;
  d_morton_code = nullptr;
  d_sorted_index = nullptr;
  d_blocks = nullptr;
  d_chunk_aabb = nullptr;
  d_active_chunks = nullptr;
  d_block_max_vals = nullptr;
  d_block_max_indices = nullptr;
  d_argmax_result = nullptr;
  h_sampled_indices = nullptr;

  num_points_ = 0;
  num_blocks_ = 0;
  num_chunks_ = 0;
}

// =============================================================================
// FpsGPU包装类实现
// =============================================================================

FpsGPU::FpsGPU() : impl_(new FpsGPUImpl()) {}

FpsGPU::~FpsGPU() {
  delete static_cast<FpsGPUImpl*>(impl_);
}

bool FpsGPU::initialize(const std::vector<Point3D>& points, const SpaceRange& range) {
  return static_cast<FpsGPUImpl*>(impl_)->initialize(points, range);
}

std::vector<size_t> FpsGPU::sample(size_t sample_count) {
  return static_cast<FpsGPUImpl*>(impl_)->sample(sample_count);
}

void FpsGPU::release() {
  static_cast<FpsGPUImpl*>(impl_)->release();
}

// =============================================================================
// 主入口函数
// =============================================================================

std::vector<size_t> yuezu_fps_gpu(const std::vector<Point3D>& point_cloud,
                                   size_t sample_count, const SpaceRange& range) {
  FpsGPU fps;
  if (!fps.initialize(point_cloud, range)) {
    return {};
  }
  return fps.sample(sample_count);
}
