#include "yuezu_fps_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

// =============================================================================
// 内部辅助函数：计算需要的位数
// =============================================================================
static inline uint32_t _calc_bits(uint32_t blocks)
{
    if (blocks <= 1)
        return 1;
    uint32_t bits = 0;
    while ((1u << bits) < blocks)
    {
        ++bits;
    }
    return bits;
}

// =============================================================================
// 修改后的构造函数
// =============================================================================

SpaceRange make_uniform_range(float min_x, float max_x, float min_y,
                              float max_y, float min_z, float max_z,
                              uint32_t blocks_per_dim)
{
    float block_size_x = (max_x - min_x) / static_cast<float>(blocks_per_dim);
    float block_size_y = (max_y - min_y) / static_cast<float>(blocks_per_dim);
    float block_size_z = (max_z - min_z) / static_cast<float>(blocks_per_dim);

    float inv_bs_x = 1.0f / block_size_x;
    float inv_bs_y = 1.0f / block_size_y;
    float inv_bs_z = 1.0f / block_size_z;

    uint32_t bits = _calc_bits(blocks_per_dim);

    return {
        min_x, max_x,
        min_y, max_y,
        min_z, max_z,
        blocks_per_dim, blocks_per_dim, blocks_per_dim,
        inv_bs_x, inv_bs_y, inv_bs_z,
        bits, bits, bits};
}

SpaceRange make_range(float min_x, float max_x, float min_y, float max_y,
                      float min_z, float max_z, uint32_t x_blocks,
                      uint32_t y_blocks, uint32_t z_blocks)
{
    float inv_bs_x = 1.0f / ((max_x - min_x) / static_cast<float>(x_blocks));
    float inv_bs_y = 1.0f / ((max_y - min_y) / static_cast<float>(y_blocks));
    float inv_bs_z = 1.0f / ((max_z - min_z) / static_cast<float>(z_blocks));

    uint32_t xb = _calc_bits(x_blocks);
    uint32_t yb = _calc_bits(y_blocks);
    uint32_t zb = _calc_bits(z_blocks);

    return {
        min_x, max_x,
        min_y, max_y,
        min_z, max_z,
        x_blocks, y_blocks, z_blocks,
        inv_bs_x, inv_bs_y, inv_bs_z,
        xb, yb, zb};
}

SpaceRange compute_bbox(const std::vector<Point3D> &points,
                        uint32_t blocks_per_dim)
{
    return compute_bbox(points, blocks_per_dim, blocks_per_dim, blocks_per_dim);
}

SpaceRange compute_bbox(const std::vector<Point3D> &points, uint32_t x_blocks,
                        uint32_t y_blocks, uint32_t z_blocks)
{
    if (points.empty())
    {
        return make_range(0, 1, 0, 1, 0, 1, 1, 1, 1);
    }

    float min_x = points[0].x, max_x = points[0].x;
    float min_y = points[0].y, max_y = points[0].y;
    float min_z = points[0].z, max_z = points[0].z;

    for (const auto &p : points)
    {
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

struct ChunkAABB
{
    float min_x, max_x;
    float min_y, max_y;
    float min_z, max_z;
};

// =============================================================================
// FpsGPU内部实现类
// =============================================================================

class FpsGPUImpl
{
public:
    FpsGPUImpl()
        : num_points_(0), d_x(nullptr), d_y(nullptr), d_z(nullptr),
          d_original_index(nullptr), d_min_dist(nullptr), d_morton_code(nullptr),
          d_sorted_index(nullptr), d_chunk_aabb(nullptr), d_active_chunks(nullptr),
          d_block_max_vals(nullptr), d_block_max_indices(nullptr),
          d_argmax_result(nullptr), d_sampled_indices(nullptr),
          d_current_sample_pos(nullptr), d_current_max_dist(nullptr),
          d_point_grid(nullptr), d_range_params(nullptr),
          d_x_lut_(nullptr), d_y_lut_(nullptr), d_z_lut_(nullptr) {}

    ~FpsGPUImpl()
    {
        release();
    }

    bool initialize(const std::vector<Point3D> &points, const SpaceRange &range);
    std::vector<size_t> sample(size_t sample_count);
    void release();

private:
    uint32_t num_points_;
    SpaceRange range_;

    float *d_x;
    float *d_y;
    float *d_z;
    uint32_t *d_original_index;
    float *d_min_dist;
    uint32_t *d_morton_code;
    uint32_t *d_sorted_index;
    ChunkAABB *d_chunk_aabb;
    int32_t *d_active_chunks;
    float *d_block_max_vals;
    uint32_t *d_block_max_indices;

    uint32_t *d_argmax_result;
    uint32_t *d_sampled_indices;
    float *d_current_sample_pos;
    float *d_current_max_dist;
    uint32_t *d_range_params;

    uint32_t num_chunks_;

    uint32_t *d_point_grid;

    std::vector<uint32_t> h_x_lut_;
    std::vector<uint32_t> h_y_lut_;
    std::vector<uint32_t> h_z_lut_;
    uint32_t *d_x_lut_;
    uint32_t *d_y_lut_;
    uint32_t *d_z_lut_;
};

__device__ __forceinline__ uint32_t pack_grid(uint32_t bx, uint32_t by, uint32_t bz)
{
    return (bx << 20) | (by << 10) | bz;
}

__device__ __forceinline__ uint32_t unpack_bx(uint32_t packed)
{
    return (packed >> 20) & 0x3ff;
}

__device__ __forceinline__ uint32_t unpack_by(uint32_t packed)
{
    return (packed >> 10) & 0x3ff;
}

__device__ __forceinline__ uint32_t unpack_bz(uint32_t packed)
{
    return packed & 0x3ff;
}

// =============================================================================
// GPU内核：基础组件
// =============================================================================

__global__ void init_min_dist_kernel(float *__restrict__ min_dist, uint32_t num_points)
{
    uint32_t idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= num_points)
        return;
    min_dist[idx] = INF_DISTANCE;
}

__global__ void update_first_point_kernel_ptr(
    const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
    float *__restrict__ min_dist,
    const float *__restrict__ q_pos_ptr,
    uint32_t num_points)
{

    const float qx = q_pos_ptr[0];
    const float qy = q_pos_ptr[1];
    const float qz = q_pos_ptr[2];

    uint32_t idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= num_points)
        return;

    float dx = x[idx] - qx;
    float dy = y[idx] - qy;
    float dz = z[idx] - qz;

    min_dist[idx] = dx * dx + dy * dy + dz * dz;
}

__global__ void build_chunk_aabb_kernel(
    const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
    const uint32_t *__restrict__ sorted_indices,
    ChunkAABB *__restrict__ aabbs, uint32_t num_points)
{

    const int block_size = 256;
    __shared__ float sh_min_x[256];
    __shared__ float sh_max_x[256];
    __shared__ float sh_min_y[256];
    __shared__ float sh_max_y[256];
    __shared__ float sh_min_z[256];
    __shared__ float sh_max_z[256];

    uint32_t chunk_idx = blockIdx.x;
    uint32_t start = chunk_idx * CHUNK_SIZE;
    uint32_t end = min(start + CHUNK_SIZE, num_points);

    if (start >= num_points)
        return;

    int tid = threadIdx.x;
    if (tid < block_size)
    {
        sh_min_x[tid] = INF_DISTANCE;
        sh_max_x[tid] = -INF_DISTANCE;
        sh_min_y[tid] = INF_DISTANCE;
        sh_max_y[tid] = -INF_DISTANCE;
        sh_min_z[tid] = INF_DISTANCE;
        sh_max_z[tid] = -INF_DISTANCE;
    }
    __syncthreads();

    float min_x = INF_DISTANCE, max_x = -INF_DISTANCE;
    float min_y = INF_DISTANCE, max_y = -INF_DISTANCE;
    float min_z = INF_DISTANCE, max_z = -INF_DISTANCE;

    for (uint32_t i = start + tid; i < end; i += block_size)
    {
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

    if (tid < block_size)
    {
        sh_min_x[tid] = min_x;
        sh_max_x[tid] = max_x;
        sh_min_y[tid] = min_y;
        sh_max_y[tid] = max_y;
        sh_min_z[tid] = min_z;
        sh_max_z[tid] = max_z;
    }
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sh_min_x[tid] = min(sh_min_x[tid], sh_min_x[tid + s]);
            sh_max_x[tid] = max(sh_max_x[tid], sh_max_x[tid + s]);
            sh_min_y[tid] = min(sh_min_y[tid], sh_min_y[tid + s]);
            sh_max_y[tid] = max(sh_max_y[tid], sh_max_y[tid + s]);
            sh_min_z[tid] = min(sh_min_z[tid], sh_min_z[tid + s]);
            sh_max_z[tid] = max(sh_max_z[tid], sh_max_z[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        aabbs[chunk_idx].min_x = sh_min_x[0];
        aabbs[chunk_idx].max_x = sh_max_x[0];
        aabbs[chunk_idx].min_y = sh_min_y[0];
        aabbs[chunk_idx].max_y = sh_max_y[0];
        aabbs[chunk_idx].min_z = sh_min_z[0];
        aabbs[chunk_idx].max_z = sh_max_z[0];
    }
}

// =============================================================================
// 高效 ArgMax - 展开归约 + warp 优化
// =============================================================================

__global__ void block_argmax_kernel_v6(const float *__restrict__ data, uint32_t num_points,
                                       float *__restrict__ block_max_vals, uint32_t *__restrict__ block_max_indices)
{
    const int block_size = 1024;
    __shared__ float sh_max[1024];
    __shared__ uint32_t sh_idx[1024];

    int tid = threadIdx.x;
    int block_id = blockIdx.x;

    if (tid < 1024)
    {
        sh_max[tid] = -INF_DISTANCE;
        sh_idx[tid] = 0;
    }
    __syncthreads();

    float max_val = -INF_DISTANCE;
    uint32_t max_idx = 0;

    for (uint32_t i = block_id * block_size + tid; i < num_points; i += gridDim.x * block_size)
    {
        float val = data[i];
        if (val > max_val)
        {
            max_val = val;
            max_idx = i;
        }
    }

    if (tid < 1024)
    {
        sh_max[tid] = max_val;
        sh_idx[tid] = max_idx;
    }
    __syncthreads();

    if (tid < 512)
    {
        if (sh_max[tid] < sh_max[tid + 512])
        {
            sh_max[tid] = sh_max[tid + 512];
            sh_idx[tid] = sh_idx[tid + 512];
        }
    }
    __syncthreads();

    if (tid < 256)
    {
        if (sh_max[tid] < sh_max[tid + 256])
        {
            sh_max[tid] = sh_max[tid + 256];
            sh_idx[tid] = sh_idx[tid + 256];
        }
    }
    __syncthreads();

    if (tid < 128)
    {
        if (sh_max[tid] < sh_max[tid + 128])
        {
            sh_max[tid] = sh_max[tid + 128];
            sh_idx[tid] = sh_idx[tid + 128];
        }
    }
    __syncthreads();

    if (tid < 64)
    {
        if (sh_max[tid] < sh_max[tid + 64])
        {
            sh_max[tid] = sh_max[tid + 64];
            sh_idx[tid] = sh_idx[tid + 64];
        }
    }
    __syncthreads();

    if (tid < 32)
    {
        volatile float *v_max = sh_max;
        volatile uint32_t *v_idx = sh_idx;

        if (v_max[tid] < v_max[tid + 32])
        {
            v_max[tid] = v_max[tid + 32];
            v_idx[tid] = v_idx[tid + 32];
        }
        if (v_max[tid] < v_max[tid + 16])
        {
            v_max[tid] = v_max[tid + 16];
            v_idx[tid] = v_idx[tid + 16];
        }
        if (v_max[tid] < v_max[tid + 8])
        {
            v_max[tid] = v_max[tid + 8];
            v_idx[tid] = v_idx[tid + 8];
        }
        if (v_max[tid] < v_max[tid + 4])
        {
            v_max[tid] = v_max[tid + 4];
            v_idx[tid] = v_idx[tid + 4];
        }
        if (v_max[tid] < v_max[tid + 2])
        {
            v_max[tid] = v_max[tid + 2];
            v_idx[tid] = v_idx[tid + 2];
        }
        if (v_max[tid] < v_max[tid + 1])
        {
            v_max[tid] = v_max[tid + 1];
            v_idx[tid] = v_idx[tid + 1];
        }
    }

    if (tid == 0)
    {
        block_max_vals[block_id] = sh_max[0];
        block_max_indices[block_id] = sh_idx[0];
    }
}

__global__ void final_argmax_kernel_v6(const float *__restrict__ block_vals, const uint32_t *__restrict__ block_indices,
                                       uint32_t num_blocks, uint32_t *__restrict__ result)
{
    const int block_size = 128;
    __shared__ float sh_max[128];
    __shared__ uint32_t sh_idx[128];

    int tid = threadIdx.x;

    if (tid < 128)
    {
        sh_max[tid] = -INF_DISTANCE;
        sh_idx[tid] = 0;
    }
    __syncthreads();

    float max_val = -INF_DISTANCE;
    uint32_t max_idx = 0;

    for (uint32_t i = tid; i < num_blocks; i += block_size)
    {
        float val = block_vals[i];
        if (val > max_val)
        {
            max_val = val;
            max_idx = block_indices[i];
        }
    }

    if (tid < 128)
    {
        sh_max[tid] = max_val;
        sh_idx[tid] = max_idx;
    }
    __syncthreads();

    if (tid < 64)
    {
        if (sh_max[tid] < sh_max[tid + 64])
        {
            sh_max[tid] = sh_max[tid + 64];
            sh_idx[tid] = sh_idx[tid + 64];
        }
    }
    __syncthreads();

    if (tid < 32)
    {
        volatile float *v_max = sh_max;
        volatile uint32_t *v_idx = sh_idx;

        if (v_max[tid] < v_max[tid + 32])
        {
            v_max[tid] = v_max[tid + 32];
            v_idx[tid] = v_idx[tid + 32];
        }
        if (v_max[tid] < v_max[tid + 16])
        {
            v_max[tid] = v_max[tid + 16];
            v_idx[tid] = v_idx[tid + 16];
        }
        if (v_max[tid] < v_max[tid + 8])
        {
            v_max[tid] = v_max[tid + 8];
            v_idx[tid] = v_idx[tid + 8];
        }
        if (v_max[tid] < v_max[tid + 4])
        {
            v_max[tid] = v_max[tid + 4];
            v_idx[tid] = v_idx[tid + 4];
        }
        if (v_max[tid] < v_max[tid + 2])
        {
            v_max[tid] = v_max[tid + 2];
            v_idx[tid] = v_idx[tid + 2];
        }
        if (v_max[tid] < v_max[tid + 1])
        {
            v_max[tid] = v_max[tid + 1];
            v_idx[tid] = v_idx[tid + 1];
        }
    }

    if (tid == 0)
    {
        *result = sh_idx[0];
    }
}

// =============================================================================
// 辅助Kernel
// =============================================================================

__global__ void gather_sample_data_kernel(
    const uint32_t *__restrict__ idx_ptr,
    const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
    const float *__restrict__ min_dist,
    float *__restrict__ out_pos, float *__restrict__ out_dist)
{

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        uint32_t idx = *idx_ptr;
        out_pos[0] = x[idx];
        out_pos[1] = y[idx];
        out_pos[2] = z[idx];
        *out_dist = min_dist[idx];
    }
}

__global__ void scatter_index_kernel(
    const uint32_t *__restrict__ idx_ptr,
    const uint32_t *__restrict__ d_original_index,
    uint32_t *__restrict__ output_array,
    size_t output_pos)
{

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        uint32_t idx = *idx_ptr;
        uint32_t orig_idx = d_original_index[idx];
        output_array[output_pos] = orig_idx;
    }
}

__global__ void argmax_postprocess_kernel(
    const float *__restrict__ block_vals,
    const uint32_t *__restrict__ block_indices,
    uint32_t num_blocks,

    // 原始数据
    const float *__restrict__ x,
    const float *__restrict__ y,
    const float *__restrict__ z,
    const float *__restrict__ min_dist,
    const uint32_t *__restrict__ original_index,

    // 输出
    uint32_t *__restrict__ sampled_indices,
    size_t output_pos,

    float *__restrict__ out_pos,    // 3
    float *__restrict__ out_dist,   // 1
    uint32_t *__restrict__ out_idx, // argmax idx

    // range compute
    float inv_bs_x, float inv_bs_y, float inv_bs_z,
    float min_x, float min_y, float min_z,
    uint32_t x_blocks, uint32_t y_blocks, uint32_t z_blocks,
    uint32_t *__restrict__ range_params)
{
    const int block_size = 128;

    __shared__ float sh_max[128];
    __shared__ uint32_t sh_idx[128];

    int tid = threadIdx.x;

    float max_val = -INF_DISTANCE;
    uint32_t max_idx = 0;

    // === step1: reduce block argmax ===
    for (uint32_t i = tid; i < num_blocks; i += block_size)
    {
        float val = block_vals[i];
        if (val > max_val)
        {
            max_val = val;
            max_idx = block_indices[i];
        }
    }

    sh_max[tid] = max_val;
    sh_idx[tid] = max_idx;
    __syncthreads();

    // reduction
    for (int s = 64; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (sh_max[tid] < sh_max[tid + s])
            {
                sh_max[tid] = sh_max[tid + s];
                sh_idx[tid] = sh_idx[tid + s];
            }
        }
        __syncthreads();
    }

    // === step2: 单线程做后处理 ===
    if (tid == 0)
    {
        uint32_t idx = sh_idx[0];
        *out_idx = idx;

        // gather pos
        float qx = x[idx];
        float qy = y[idx];
        float qz = z[idx];

        out_pos[0] = qx;
        out_pos[1] = qy;
        out_pos[2] = qz;

        float dist = min_dist[idx];
        *out_dist = dist;

        // scatter index
        sampled_indices[output_pos] = original_index[idx];

        // === compute range ===
        float radius = sqrtf(dist);

        int nx = (int)ceilf(radius * inv_bs_x) + 1;
        int ny = (int)ceilf(radius * inv_bs_y) + 1;
        int nz = (int)ceilf(radius * inv_bs_z) + 1;

        uint32_t cbx = (uint32_t)((qx - min_x) * inv_bs_x);
        uint32_t cby = (uint32_t)((qy - min_y) * inv_bs_y);
        uint32_t cbz = (uint32_t)((qz - min_z) * inv_bs_z);

        cbx = min(cbx, x_blocks - 1);
        cby = min(cby, y_blocks - 1);
        cbz = min(cbz, z_blocks - 1);

        range_params[0] = max(0, (int)cbx - nx);
        range_params[1] = min((int)x_blocks - 1, (int)cbx + nx);
        range_params[2] = max(0, (int)cby - ny);
        range_params[3] = min((int)y_blocks - 1, (int)cby + ny);
        range_params[4] = max(0, (int)cbz - nz);
        range_params[5] = min((int)z_blocks - 1, (int)cbz + nz);
    }
}

// =============================================================================
// GPU Kernel：计算莫顿码 + 保存网格索引
// =============================================================================
__global__ void compute_morton_and_grid_indices_kernel(
    const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
    uint32_t *__restrict__ morton_codes, uint32_t *__restrict__ indices,
    uint32_t *__restrict__ out_grid,
    float min_x, float min_y, float min_z,
    float inv_block_size_x, float inv_block_size_y, float inv_block_size_z,
    uint32_t x_blocks, uint32_t y_blocks, uint32_t z_blocks,
    uint32_t num_points,
    const uint32_t *__restrict__ d_x_lut,
    const uint32_t *__restrict__ d_y_lut,
    const uint32_t *__restrict__ d_z_lut)
{

    uint32_t idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= num_points)
        return;

    float px = x[idx];
    float py = y[idx];
    float pz = z[idx];

    uint32_t bx = static_cast<uint32_t>((px - min_x) * inv_block_size_x);
    uint32_t by = static_cast<uint32_t>((py - min_y) * inv_block_size_y);
    uint32_t bz = static_cast<uint32_t>((pz - min_z) * inv_block_size_z);

    bx = min(bx, x_blocks - 1);
    by = min(by, y_blocks - 1);
    bz = min(bz, z_blocks - 1);

    out_grid[idx] = pack_grid(bx, by, bz);

    morton_codes[idx] = d_x_lut[bx] | d_y_lut[by] | d_z_lut[bz];
    indices[idx] = idx;
}

// =============================================================================
// 辅助：重排网格索引 Kernel
// =============================================================================
__global__ void permute_grid_indices_kernel(
    const uint32_t *__restrict__ sorted_indices,
    const uint32_t *__restrict__ old_grid,
    uint32_t *__restrict__ new_grid,
    uint32_t num_points)
{
    uint32_t idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= num_points)
        return;

    uint32_t original_idx = sorted_indices[idx];
    new_grid[idx] = old_grid[original_idx];
}

// =============================================================================
// 修复 compute_query_range_kernel - 直接使用 inv，不再取倒数！
// =============================================================================
__global__ void compute_query_range_kernel_v4(
    const float *__restrict__ q_pos_ptr,
    const float *__restrict__ max_dist_sq_ptr,
    float inv_block_size_x, float inv_block_size_y, float inv_block_size_z,
    float min_x, float min_y, float min_z,
    uint32_t x_blocks, uint32_t y_blocks, uint32_t z_blocks,
    uint32_t *__restrict__ out_range_params)
{

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        float qx = q_pos_ptr[0];
        float qy = q_pos_ptr[1];
        float qz = q_pos_ptr[2];
        float max_dist_sq = *max_dist_sq_ptr;

        float radius = sqrt(max_dist_sq);

        // 修复：直接用 radius * inv_block_size，而不是先取倒数再除！
        int nx = static_cast<int>(ceil(radius * inv_block_size_x)) + 1;
        int ny = static_cast<int>(ceil(radius * inv_block_size_y)) + 1;
        int nz = static_cast<int>(ceil(radius * inv_block_size_z)) + 1;

        uint32_t cbx = static_cast<uint32_t>((qx - min_x) * inv_block_size_x);
        uint32_t cby = static_cast<uint32_t>((qy - min_y) * inv_block_size_y);
        uint32_t cbz = static_cast<uint32_t>((qz - min_z) * inv_block_size_z);

        cbx = min(cbx, x_blocks - 1);
        cby = min(cby, y_blocks - 1);
        cbz = min(cbz, z_blocks - 1);

        uint32_t range_min_bx = static_cast<uint32_t>(max(0, static_cast<int>(cbx) - nx));
        uint32_t range_max_bx = static_cast<uint32_t>(min(static_cast<int>(x_blocks - 1), static_cast<int>(cbx) + nx));
        uint32_t range_min_by = static_cast<uint32_t>(max(0, static_cast<int>(cby) - ny));
        uint32_t range_max_by = static_cast<uint32_t>(min(static_cast<int>(y_blocks - 1), static_cast<int>(cby) + ny));
        uint32_t range_min_bz = static_cast<uint32_t>(max(0, static_cast<int>(cbz) - nz));
        uint32_t range_max_bz = static_cast<uint32_t>(min(static_cast<int>(z_blocks - 1), static_cast<int>(cbz) + nz));

        out_range_params[0] = range_min_bx;
        out_range_params[1] = range_max_bx;
        out_range_params[2] = range_min_by;
        out_range_params[3] = range_max_by;
        out_range_params[4] = range_min_bz;
        out_range_params[5] = range_max_bz;
    }
}

// =============================================================================
// 核心优化：空间范围查询 + 距离更新
// =============================================================================
__global__ void update_by_grid_range_kernel_v4(
    const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
    const uint32_t *__restrict__ sorted_indices,
    const uint32_t *__restrict__ point_grid,
    float *__restrict__ min_dist,
    const float *__restrict__ q_pos_ptr,
    const float *__restrict__ max_dist_sq_ptr,
    const uint32_t *__restrict__ range_params,
    uint32_t num_points)
{

    const float qx = q_pos_ptr[0];
    const float qy = q_pos_ptr[1];
    const float qz = q_pos_ptr[2];

    uint32_t idx = blockIdx.x * 128 + threadIdx.x;
    if (idx >= num_points)
        return;

    uint32_t grid = __ldg(&point_grid[idx]);
    uint32_t bx = unpack_bx(grid);
    uint32_t by = unpack_by(grid);
    uint32_t bz = unpack_bz(grid);

    uint32_t range_min_bx = range_params[0];
    uint32_t range_max_bx = range_params[1];
    uint32_t range_min_by = range_params[2];
    uint32_t range_max_by = range_params[3];
    uint32_t range_min_bz = range_params[4];
    uint32_t range_max_bz = range_params[5];

    bool in_range = (bx >= range_min_bx && bx <= range_max_bx) &&
                    (by >= range_min_by && by <= range_max_by) &&
                    (bz >= range_min_bz && bz <= range_max_bz);

    if (!in_range)
    {
        return;
    }

    uint32_t real_idx = __ldg(&sorted_indices[idx]);

    float px = __ldg(&x[real_idx]);
    float py = __ldg(&y[real_idx]);
    float pz = __ldg(&z[real_idx]);

    float dx = px - qx;
    float dy = py - qy;
    float dz = pz - qz;
    float dist_sq = dx * dx + dy * dy + dz * dz;

    if (dist_sq < min_dist[real_idx])
    {
        min_dist[real_idx] = dist_sq;
    }
}

// =============================================================================
// 辅助函数
// =============================================================================

static void _compute_bit_positions(uint32_t xb, uint32_t yb, uint32_t zb,
                                   std::vector<uint32_t> &x_pos,
                                   std::vector<uint32_t> &y_pos,
                                   std::vector<uint32_t> &z_pos)
{
    uint32_t bit = 0;
    uint32_t common = std::min({xb, yb, zb});
    uint32_t maxb = std::max({xb, yb, zb});

    x_pos.clear();
    y_pos.clear();
    z_pos.clear();

    for (uint32_t i = 0; i < common; ++i)
    {
        x_pos.push_back(bit++);
        y_pos.push_back(bit++);
        z_pos.push_back(bit++);
    }

    for (uint32_t i = common; i < maxb; ++i)
    {
        if (i < xb)
            x_pos.push_back(bit++);
        if (i < yb)
            y_pos.push_back(bit++);
        if (i < zb)
            z_pos.push_back(bit++);
    }
}

static void _generate_morton_lut(uint32_t bits, const std::vector<uint32_t> &pos_list, std::vector<uint32_t> &lut)
{
    uint32_t count = 1u << bits;
    lut.resize(count);
    for (uint32_t v = 0; v < count; ++v)
    {
        uint32_t code = 0;
        for (uint32_t k = 0; k < bits; ++k)
        {
            if (v & (1u << k))
            {
                code |= 1u << pos_list[k];
            }
        }
        lut[v] = code;
    }
}

// =============================================================================
// FpsGPUImpl实现
// =============================================================================

bool FpsGPUImpl::initialize(const std::vector<Point3D> &points, const SpaceRange &range)
{
    num_points_ = static_cast<uint32_t>(points.size());
    range_ = range;

    if (num_points_ == 0)
        return false;

    cudaMalloc(&d_x, num_points_ * sizeof(float));
    cudaMalloc(&d_y, num_points_ * sizeof(float));
    cudaMalloc(&d_z, num_points_ * sizeof(float));
    cudaMalloc(&d_original_index, num_points_ * sizeof(uint32_t));
    cudaMalloc(&d_min_dist, num_points_ * sizeof(float));
    cudaMalloc(&d_morton_code, num_points_ * sizeof(uint32_t));
    cudaMalloc(&d_sorted_index, num_points_ * sizeof(uint32_t));

    std::vector<uint32_t> x_pos, y_pos, z_pos;
    _compute_bit_positions(range.x_bits_, range.y_bits_, range.z_bits_, x_pos, y_pos, z_pos);
    _generate_morton_lut(range.x_bits_, x_pos, h_x_lut_);
    _generate_morton_lut(range.y_bits_, y_pos, h_y_lut_);
    _generate_morton_lut(range.z_bits_, z_pos, h_z_lut_);

    cudaMalloc(&d_x_lut_, h_x_lut_.size() * sizeof(uint32_t));
    cudaMalloc(&d_y_lut_, h_y_lut_.size() * sizeof(uint32_t));
    cudaMalloc(&d_z_lut_, h_z_lut_.size() * sizeof(uint32_t));

    cudaMemcpy(d_x_lut_, h_x_lut_.data(), h_x_lut_.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_lut_, h_y_lut_.data(), h_y_lut_.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z_lut_, h_z_lut_.data(), h_z_lut_.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    std::vector<float> h_x(num_points_), h_y(num_points_), h_z(num_points_);
    std::vector<uint32_t> h_original_index(num_points_);
    for (uint32_t i = 0; i < num_points_; ++i)
    {
        h_x[i] = points[i].x;
        h_y[i] = points[i].y;
        h_z[i] = points[i].z;
        h_original_index[i] = static_cast<uint32_t>(points[i].original_index);
    }
    cudaMemcpy(d_x, h_x.data(), num_points_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), num_points_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z.data(), num_points_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_original_index, h_original_index.data(), num_points_ * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t *d_temp_grid;
    cudaMalloc(&d_temp_grid, num_points_ * sizeof(uint32_t));
    cudaMalloc(&d_point_grid, num_points_ * sizeof(uint32_t));

    float inv_block_size_x = range.inv_bs_x;
    float inv_block_size_y = range.inv_bs_y;
    float inv_block_size_z = range.inv_bs_z;

    uint32_t blocks = (num_points_ + 255) / 256;

    compute_morton_and_grid_indices_kernel<<<blocks, 256>>>(
        d_x, d_y, d_z, d_morton_code, d_sorted_index,
        d_temp_grid,
        range.min_x, range.min_y, range.min_z,
        inv_block_size_x, inv_block_size_y, inv_block_size_z,
        range.x_blocks, range.y_blocks, range.z_blocks, num_points_,
        d_x_lut_, d_y_lut_, d_z_lut_);

    thrust::device_ptr<uint32_t> dev_morton(d_morton_code);
    thrust::device_ptr<uint32_t> dev_index(d_sorted_index);
    thrust::sort_by_key(dev_morton, dev_morton + num_points_, dev_index);

    permute_grid_indices_kernel<<<blocks, 256>>>(
        d_sorted_index,
        d_temp_grid,
        d_point_grid,
        num_points_);

    cudaFree(d_temp_grid);

    num_chunks_ = (num_points_ + CHUNK_SIZE - 1) / CHUNK_SIZE;
    cudaMalloc(&d_chunk_aabb, num_chunks_ * sizeof(ChunkAABB));
    build_chunk_aabb_kernel<<<num_chunks_, 256>>>(
        d_x, d_y, d_z, d_sorted_index, d_chunk_aabb, num_points_);

    cudaMalloc(&d_active_chunks, num_chunks_ * sizeof(int32_t));
    const uint32_t argmax_num_blocks = 32;
    cudaMalloc(&d_block_max_vals, argmax_num_blocks * sizeof(float));
    cudaMalloc(&d_block_max_indices, argmax_num_blocks * sizeof(uint32_t));

    cudaMalloc(&d_argmax_result, sizeof(uint32_t));
    cudaMalloc(&d_sampled_indices, num_points_ * sizeof(uint32_t));
    cudaMalloc(&d_current_sample_pos, 3 * sizeof(float));
    cudaMalloc(&d_current_max_dist, sizeof(float));
    cudaMalloc(&d_range_params, 6 * sizeof(uint32_t));

    return true;
}

std::vector<size_t> FpsGPUImpl::sample(size_t sample_count)
{
    if (num_points_ == 0 || sample_count == 0)
        return {};
    sample_count = std::min(sample_count, static_cast<size_t>(num_points_));

    const uint32_t argmax_num_blocks = 32;
    uint32_t blocks = (num_points_ + 255) / 256;

    init_min_dist_kernel<<<blocks, 256>>>(d_min_dist, num_points_);

    uint32_t h_first_idx = 0;
    cudaMemcpy(d_argmax_result, &h_first_idx, sizeof(uint32_t), cudaMemcpyHostToDevice);

    gather_sample_data_kernel<<<1, 32>>>(
        d_argmax_result, d_x, d_y, d_z, d_min_dist,
        d_current_sample_pos, d_current_max_dist);

    scatter_index_kernel<<<1, 32>>>(
        d_argmax_result, d_original_index, d_sampled_indices, 0);

    update_first_point_kernel_ptr<<<blocks, 256>>>(
        d_x, d_y, d_z, d_min_dist,
        d_current_sample_pos,
        num_points_);

    for (size_t i = 1; i < sample_count; ++i)
    {
        block_argmax_kernel_v6<<<argmax_num_blocks, 1024>>>(
            d_min_dist, num_points_, d_block_max_vals, d_block_max_indices);

        argmax_postprocess_kernel<<<1, 128>>>(
            d_block_max_vals,
            d_block_max_indices,
            argmax_num_blocks,

            d_x, d_y, d_z,
            d_min_dist,
            d_original_index,

            d_sampled_indices,
            i,

            d_current_sample_pos,
            d_current_max_dist,
            d_argmax_result,

            range_.inv_bs_x, range_.inv_bs_y, range_.inv_bs_z,
            range_.min_x, range_.min_y, range_.min_z,
            range_.x_blocks, range_.y_blocks, range_.z_blocks,
            d_range_params);

        uint32_t update_blocks = (num_points_ + 128 - 1) / 128;

        update_by_grid_range_kernel_v4<<<update_blocks, 128>>>(
            d_x, d_y, d_z,
            d_sorted_index,
            d_point_grid,
            d_min_dist,
            d_current_sample_pos,
            d_current_max_dist,
            d_range_params,
            num_points_);
    }

    std::vector<uint32_t> h_indices(sample_count);
    cudaMemcpy(h_indices.data(), d_sampled_indices, sample_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<size_t> result(sample_count);
    for (size_t i = 0; i < sample_count; ++i)
        result[i] = h_indices[i];

    return result;
}

void FpsGPUImpl::release()
{
    if (d_x)
        cudaFree(d_x);
    if (d_y)
        cudaFree(d_y);
    if (d_z)
        cudaFree(d_z);
    if (d_original_index)
        cudaFree(d_original_index);
    if (d_min_dist)
        cudaFree(d_min_dist);
    if (d_morton_code)
        cudaFree(d_morton_code);
    if (d_sorted_index)
        cudaFree(d_sorted_index);
    if (d_chunk_aabb)
        cudaFree(d_chunk_aabb);
    if (d_active_chunks)
        cudaFree(d_active_chunks);
    if (d_block_max_vals)
        cudaFree(d_block_max_vals);
    if (d_block_max_indices)
        cudaFree(d_block_max_indices);
    if (d_argmax_result)
        cudaFree(d_argmax_result);
    if (d_sampled_indices)
        cudaFree(d_sampled_indices);
    if (d_current_sample_pos)
        cudaFree(d_current_sample_pos);
    if (d_current_max_dist)
        cudaFree(d_current_max_dist);
    if (d_point_grid)
        cudaFree(d_point_grid);
    if (d_range_params)
        cudaFree(d_range_params);
    if (d_x_lut_)
        cudaFree(d_x_lut_);
    if (d_y_lut_)
        cudaFree(d_y_lut_);
    if (d_z_lut_)
        cudaFree(d_z_lut_);

    d_x = nullptr;
    d_y = nullptr;
    d_z = nullptr;
    d_original_index = nullptr;
    d_min_dist = nullptr;
    d_morton_code = nullptr;
    d_sorted_index = nullptr;
    d_chunk_aabb = nullptr;
    d_active_chunks = nullptr;
    d_block_max_vals = nullptr;
    d_block_max_indices = nullptr;
    d_argmax_result = nullptr;
    d_sampled_indices = nullptr;
    d_current_sample_pos = nullptr;
    d_current_max_dist = nullptr;
    d_point_grid = nullptr;
    d_range_params = nullptr;
    d_x_lut_ = nullptr;
    d_y_lut_ = nullptr;
    d_z_lut_ = nullptr;

    h_x_lut_.clear();
    h_y_lut_.clear();
    h_z_lut_.clear();

    num_points_ = 0;
    num_chunks_ = 0;
}

FpsGPU::FpsGPU() : impl_(new FpsGPUImpl()) {}
FpsGPU::~FpsGPU() { delete static_cast<FpsGPUImpl *>(impl_); }
bool FpsGPU::initialize(const std::vector<Point3D> &points, const SpaceRange &range) { return static_cast<FpsGPUImpl *>(impl_)->initialize(points, range); }
std::vector<size_t> FpsGPU::sample(size_t sample_count) { return static_cast<FpsGPUImpl *>(impl_)->sample(sample_count); }
void FpsGPU::release() { static_cast<FpsGPUImpl *>(impl_)->release(); }

std::vector<size_t> yuezu_fps_gpu(const std::vector<Point3D> &point_cloud,
                                  size_t sample_count, const SpaceRange &range)
{
    FpsGPU fps;
    if (!fps.initialize(point_cloud, range))
        return {};
    return fps.sample(sample_count);
}
