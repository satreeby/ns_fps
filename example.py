import numpy as np
import yuezu_fps.yuezu_fps_module as yf


# =============================================================================
# 方式1：手动指定范围 + 粒度（已知点云范围时使用）
# =============================================================================

def fps_manual(points, n_samples, 
               min_x, max_x, min_y, max_y, min_z, max_z,
               x_blocks=16, y_blocks=16, z_blocks=16):
    """
    手动指定空间范围和粒度进行FPS采样
    
    适用于：已知点云范围，想精确控制空间分割粒度
    """
    # 创建 SpaceRange
    space_range = yf.make_range(
        float(min_x), float(max_x),
        float(min_y), float(max_y),
        float(min_z), float(max_z),
        x_blocks, y_blocks, z_blocks
    )
    
    # 执行 FPS
    return yf.fps(points, n_samples, space_range)


# =============================================================================
# 方式2：自适应计算范围（便捷辅助函数）
# =============================================================================

def fps_auto(points, n_samples, x_blocks=16, y_blocks=16, z_blocks=16):
    """
    自动计算点云范围，使用指定粒度进行FPS采样
    
    适用于：快速使用，不关心精确范围
    """
    # 自动统计范围
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    min_z, max_z = points[:, 2].min(), points[:, 2].max()
    
    # 添加边距
    eps = 1e-4
    return fps_manual(points, n_samples,
                      min_x - eps, max_x + eps,
                      min_y - eps, max_y + eps,
                      min_z - eps, max_z + eps,
                      x_blocks, y_blocks, z_blocks)


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    # 生成测试点云
    np.random.seed(42)
    points = np.random.randn(10000, 3).astype(np.float32) * 50
    n_samples = 1000
    
    print(f"点云: {len(points)} 点")
    print(f"实际范围: X[{points[:,0].min():.1f},{points[:,0].max():.1f}], "
          f"Y[{points[:,1].min():.1f},{points[:,1].max():.1f}], "
          f"Z[{points[:,2].min():.1f},{points[:,2].max():.1f}]")
    
    # ---------------------------------------------------------
    # 示例1：手动指定范围 + 粒度（推荐，已知范围时使用）
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("示例1：手动指定范围和粒度")
    print("=" * 60)
    
    # 已知点云范围是 [-200, 200] x [-100, 100] x [-50, 50]
    # 配置粒度：X细(32块), Y中(16块), Z粗(8块)
    indices = fps_manual(
        points, n_samples,
        min_x=-200, max_x=200,
        min_y=-100, max_y=100,
        min_z=-50, max_z=50,
        x_blocks=32, y_blocks=16, z_blocks=8
    )
    
    print(f"配置范围: [-200,200]×[-100,100]×[-50,50]")
    print(f"配置粒度: 32×16×8 = {32*16*8} 块")
    print(f"采样结果: {len(indices)} 点")
    print(f"前10个索引: {indices[:10]}")
    
    # ---------------------------------------------------------
    # 示例2：自适应计算范围（快速使用）
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("示例2：自适应计算范围")
    print("=" * 60)
    
    indices = fps_auto(points, n_samples, x_blocks=16, y_blocks=16, z_blocks=16)
    
    print(f"自动计算范围并采样")
    print(f"采样结果: {len(indices)} 点")
    
    # ---------------------------------------------------------
    # 示例3：直接使用底层 API（最灵活）
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("示例3：直接使用底层 API")
    print("=" * 60)
    
    # 创建 SpaceRange 对象
    space_range = yf.make_range(-150, 150, -150, 150, -150, 150, 16, 16, 16)
    
    # 查看配置
    print(f"SpaceRange: X[{space_range.min_x},{space_range.max_x}], "
          f"Y[{space_range.min_y},{space_range.max_y}], "
          f"Z[{space_range.min_z},{space_range.max_z}]")
    print(f"粒度: {space_range.x_blocks}×{space_range.y_blocks}×{space_range.z_blocks}")
    print(f"编码: {space_range.total_bits()} bits")
    
    # 执行 FPS
    indices = yf.fps(points, n_samples, space_range)
    print(f"采样结果: {len(indices)} 点")