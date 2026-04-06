import numpy as np
import yuezu_fps.yuezu_fps_module as yf


# =============================================================================
# Method 1: Manually Specify Range + Granularity (Use when point cloud range is known)
# =============================================================================

def fps_manual(points, n_samples, 
               min_x, max_x, min_y, max_y, min_z, max_z,
               x_blocks=16, y_blocks=16, z_blocks=16):
    """
    Perform FPS sampling by manually specifying spatial range and granularity
    
    Applicable when: Point cloud range is known, and precise control over spatial segmentation granularity is desired
    """
    # Create SpaceRange
    space_range = yf.make_range(
        float(min_x), float(max_x),
        float(min_y), float(max_y),
        float(min_z), float(max_z),
        x_blocks, y_blocks, z_blocks
    )
    
    # Execute FPS
    return yf.fps(points, n_samples, space_range)


# =============================================================================
# Method 2: Adaptive Range Calculation (Convenience Helper Function)
# =============================================================================

def fps_auto(points, n_samples, x_blocks=16, y_blocks=16, z_blocks=16):
    """
    Automatically calculate point cloud range and perform FPS sampling with specified granularity
    
    Applicable when: Quick use is needed, and precise range is not a concern
    """
    # Automatically calculate range
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    min_z, max_z = points[:, 2].min(), points[:, 2].max()
    
    # Add margin
    eps = 1e-4
    return fps_manual(points, n_samples,
                      min_x - eps, max_x + eps,
                      min_y - eps, max_y + eps,
                      min_z - eps, max_z + eps,
                      x_blocks, y_blocks, z_blocks)


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    # Generate test point cloud
    np.random.seed(42)
    points = np.random.randn(10000, 3).astype(np.float32) * 50
    n_samples = 1000
    
    print(f"Point cloud: {len(points)} points")
    print(f"Actual range: X[{points[:,0].min():.1f},{points[:,0].max():.1f}], "
          f"Y[{points[:,1].min():.1f},{points[:,1].max():.1f}], "
          f"Z[{points[:,2].min():.1f},{points[:,2].max():.1f}]")
    
    # ---------------------------------------------------------
    # Example 1: Manually Specify Range + Granularity (Recommended, use when range is known)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 1: Manually Specify Range and Granularity")
    print("=" * 60)
    
    # Known point cloud range is [-200, 200] x [-100, 100] x [-50, 50]
    # Configure granularity: X fine (32 blocks), Y medium (16 blocks), Z coarse (8 blocks)
    indices = fps_manual(
        points, n_samples,
        min_x=-200, max_x=200,
        min_y=-100, max_y=100,
        min_z=-50, max_z=50,
        x_blocks=32, y_blocks=16, z_blocks=8
    )
    
    print(f"Configured range: [-200,200]×[-100,100]×[-50,50]")
    print(f"Configured granularity: 32×16×8 = {32*16*8} blocks")
    print(f"Sampling result: {len(indices)} points")
    print(f"First 10 indices: {indices[:10]}")
    
    # ---------------------------------------------------------
    # Example 2: Adaptive Range Calculation (Quick use)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: Adaptive Range Calculation")
    print("=" * 60)
    
    indices = fps_auto(points, n_samples, x_blocks=16, y_blocks=16, z_blocks=16)
    
    print(f"Automatically calculate range and sample")
    print(f"Sampling result: {len(indices)} points")
    
    # ---------------------------------------------------------
    # Example 3: Directly Use Low-level API (Most flexible)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 3: Directly Use Low-level API")
    print("=" * 60)
    
    # Create SpaceRange object
    space_range = yf.make_range(-150, 150, -150, 150, -150, 150, 16, 16, 16)
    
    # Check configuration
    print(f"SpaceRange: X[{space_range.min_x},{space_range.max_x}], "
          f"Y[{space_range.min_y},{space_range.max_y}], "
          f"Z[{space_range.min_z},{space_range.max_z}]")
    print(f"Granularity: {space_range.x_blocks}×{space_range.y_blocks}×{space_range.z_blocks}")
    print(f"Encoding: {space_range.total_bits()} bits")
    
    # Execute FPS
    indices = yf.fps(points, n_samples, space_range)
    print(f"Sampling result: {len(indices)} points")