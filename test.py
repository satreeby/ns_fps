import numpy as np
import time
import yuezu_fps.yuezu_fps_module as yf


# =============================================================================
# Python 端 SpaceRange 计算
# =============================================================================

def compute_bbox_python(points, x_blocks=16, y_blocks=16, z_blocks=16, margin=0.01):
    """
    在 Python 端计算 SpaceRange，确保包含所有点
    """
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    min_z, max_z = points[:, 2].min(), points[:, 2].max()
    
    # 添加边距防止边界问题
    x_pad = (max_x - min_x) * margin + 1e-4
    y_pad = (max_y - min_y) * margin + 1e-4
    z_pad = (max_z - min_z) * margin + 1e-4
    
    return yf.make_range(
        float(min_x - x_pad), float(max_x + x_pad),
        float(min_y - y_pad), float(max_y + y_pad),
        float(min_z - z_pad), float(max_z + z_pad),
        x_blocks, y_blocks, z_blocks
    )


# =============================================================================
# Python 参考实现（暴力 FPS，无空间限制）
# =============================================================================

def fps_numpy(points: np.ndarray, n_samples: int):
    """传统 FPS，遍历所有点"""
    N = len(points)
    if n_samples >= N:
        return np.arange(N), np.zeros(N)
    
    indices = np.zeros(n_samples, dtype=np.int64)
    indices[0] = 0
    
    min_distances = np.full(N, np.inf, dtype=np.float32)
    points_f = points.astype(np.float32)
    max_dists = np.zeros(n_samples, dtype=np.float32)
    max_dists[0] = np.inf
    
    for i in range(1, n_samples):
        last_point = points_f[indices[i - 1]]
        dists = np.sum((points_f - last_point) ** 2, axis=1)
        min_distances = np.minimum(min_distances, dists)
        
        max_dists[i] = np.max(min_distances)
        indices[i] = np.argmax(min_distances)
        
        if max_dists[i] == 0:
            indices = indices[:i]
            max_dists = max_dists[:i]
            break
    
    return indices, max_dists


def compute_distance_sequence(points: np.ndarray, indices: np.ndarray):
    """从索引序列计算每步最大距离"""
    N = len(points)
    n = len(indices)
    
    min_distances = np.full(N, np.inf, dtype=np.float32)
    points_f = points.astype(np.float32)
    max_dists = np.zeros(n, dtype=np.float32)
    max_dists[0] = np.inf
    
    for i in range(1, n):
        last_point = points_f[indices[i - 1]]
        dists = np.sum((points_f - last_point) ** 2, axis=1)
        min_distances = np.minimum(min_distances, dists)
        max_dists[i] = np.max(min_distances)
    
    return max_dists


# =============================================================================
# 核心验证函数
# =============================================================================

def verify_correctness():
    """验证 C++ 实现与 Python 参考实现的一致性"""
    print("=" * 70)
    print("FPS 正确性验证（Python 计算 SpaceRange）")
    print("=" * 70)
    
    np.random.seed(42)
    
    test_cases = [
        ("小规模 100点/10样", np.random.randn(100, 3).astype(np.float32), 10),
        ("中等 1k点/100样", np.random.randn(1000, 3).astype(np.float32), 100),
        ("大规模 10k点/1k样", np.random.randn(10000, 3).astype(np.float32), 1000),
        ("不均匀分布", np.random.randn(5000, 3).astype(np.float32) * [10, 1, 0.1], 500),
    ]
    
    all_passed = True
    
    for name, points, n_samples in test_cases:
        print(f"\n【{name}】")
        
        # 1. Python 端计算 SpaceRange
        range_obj = compute_bbox_python(points, 16, 16, 16)
        
        # 2. Python 暴力实现
        t0 = time.time()
        py_idx, py_dists = fps_numpy(points, n_samples)
        py_time = (time.time() - t0) * 1000
        
        # 3. C++ 实现（传入 Python 计算的 range）
        t0 = time.time()
        cpp_idx = yf.fps(points, n_samples, range_obj)
        cpp_time = (time.time() - t0) * 1000
        
        # 4. 计算 C++ 距离序列
        cpp_dists = compute_distance_sequence(points, cpp_idx)
        
        # 5. 对比
        min_len = min(len(py_dists), len(cpp_dists))
        max_diff = 0.0
        first_diff_idx = -1
        
        for i in range(min_len):
            diff = abs(py_dists[i] - cpp_dists[i])
            if diff > max_diff:
                max_diff = diff
            if diff > 1e-3 and first_diff_idx == -1:
                first_diff_idx = i
        
        length_ok = abs(len(py_dists) - len(cpp_dists)) <= 1
        dist_ok = max_diff < 1e-3
        passed = length_ok and dist_ok
        
        status = "✓ PASS" if passed else "✗ FAIL"
        
        print(f"  Python: {py_time:7.2f}ms | C++: {cpp_time:6.2f}ms | 加速{py_time / cpp_time:5.1f}x")
        print(f"  采样数: {len(py_idx)}(Py) vs {len(cpp_idx)}(C++) | 最大距离差: {max_diff:.2e} [{status}]")
        
        if not passed:
            all_passed = False
            if first_diff_idx >= 0:
                print(f"  首个差异@step{first_diff_idx}: Py={py_dists[first_diff_idx]:.6f}, C++={cpp_dists[first_diff_idx]:.6f}")
    
    print("\n" + "=" * 70)
    print(f"验证结果: {'全部通过 ✓' if all_passed else '存在失败 ✗'}")
    print("=" * 70)
    
    return all_passed


def verify_granularity():
    """验证不同空间粒度配置"""
    print("\n" + "=" * 70)
    print("空间粒度配置测试（Python 计算 SpaceRange）")
    print("=" * 70)
    
    np.random.seed(42)
    points = np.random.randn(5000, 3).astype(np.float32) * 100
    n_samples = 500
    
    # Python 参考
    py_idx, py_dists = fps_numpy(points, n_samples)
    print(f"Python参考: {len(py_dists)} samples")
    
    # 粒度配置
    configs = [
        (8, 8, 8, "粗粒度 8x8x8"),
        (16, 16, 16, "标准 16x16x16"),
        (32, 32, 32, "细粒度 32x32x32"),
        (64, 64, 64, "超细 64x64x64"),
        (16, 8, 4, "非均匀 16x8x4"),
        (4, 16, 64, "非均匀 4x16x64"),
        (32, 16, 8, "非均匀 32x16x8"),
    ]
    
    print(f"\n{'配置':<20} | {'总块数':>10} | {'Py样数':>6} | {'C++样数':>6} | {'最大距离差':>12} | {'时间(ms)':>8} | {'结果':>6}")
    print("-" * 95)
    
    all_passed = True
    
    for xb, yb, zb, desc in configs:
        # Python 计算 SpaceRange
        range_obj = compute_bbox_python(points, xb, yb, zb)
        
        t0 = time.time()
        cpp_idx = yf.fps(points, n_samples, range_obj)
        cpp_time = (time.time() - t0) * 1000
        
        cpp_dists = compute_distance_sequence(points, cpp_idx)
        
        min_len = min(len(py_dists), len(cpp_dists))
        max_diff = 0.0
        for i in range(min_len):
            diff = abs(py_dists[i] - cpp_dists[i])
            if diff > max_diff:
                max_diff = diff
        
        total_blocks = xb * yb * zb
        passed = max_diff < 1e-3 and abs(len(py_dists) - len(cpp_dists)) <= 1
        status = "✓" if passed else "✗"
        
        print(f"{desc:<20} | {total_blocks:>10} | {len(py_idx):>6} | {len(cpp_idx):>6} | "
              f"{max_diff:>12.2e} | {cpp_time:>8.2f} | {status:>6}")
        
        if not passed:
            all_passed = False
    
    print("-" * 95)
    print(f"粒度测试: {'全部通过 ✓' if all_passed else '存在失败 ✗'}")
    
    return all_passed


def verify_manual_range():
    """演示手动指定范围的风险（范围不足会导致采样不足）"""
    print("\n" + "=" * 70)
    print("手动范围测试（演示范围不足的后果）")
    print("=" * 70)
    
    np.random.seed(123)
    points = np.random.randn(3000, 3).astype(np.float32)
    points[:, 0] = points[:, 0] * 50 + 100   # X: [50, 150]
    points[:, 1] = points[:, 1] * 20 - 30    # Y: [-50, -10]
    points[:, 2] = points[:, 2] * 5 + 10     # Z: [5, 15]
    
    n_samples = 300
    
    print(f"实际数据范围: X[{points[:, 0].min():.1f},{points[:, 0].max():.1f}], "
          f"Y[{points[:, 1].min():.1f},{points[:, 1].max():.1f}], "
          f"Z[{points[:, 2].min():.1f},{points[:, 2].max():.1f}]")
    
    # 对比：正确范围 vs 错误范围
    test_cases = [
        ("正确范围（Python计算）", compute_bbox_python(points, 16, 16, 8)),
        ("手动-包含所有", yf.make_range(0, 200, -100, 0, 0, 20, 16, 16, 8)),
        ("手动-Y不足", yf.make_range(0, 200, -40, -20, 0, 20, 16, 16, 8)),  # Y太紧
        ("手动-Z不足", yf.make_range(0, 200, -100, 0, 10, 15, 16, 16, 8)),  # Z太紧
    ]
    
    print(f"\n{'范围配置':<20} | {'X范围':>16} | {'Y范围':>16} | {'Z范围':>12} | {'C++样数':>8} | {'状态':>10}")
    print("-" * 100)
    
    for label, range_obj in test_cases:
        cpp_idx = yf.fps(points, n_samples, range_obj)
        
        # 统计范围内点数
        in_range = (
            (points[:, 0] >= range_obj.min_x) & (points[:, 0] <= range_obj.max_x) &
            (points[:, 1] >= range_obj.min_y) & (points[:, 1] <= range_obj.max_y) &
            (points[:, 2] >= range_obj.min_z) & (points[:, 2] <= range_obj.max_z)
        )
        n_in_range = np.sum(in_range)
        
        if len(cpp_idx) == n_samples:
            status = "✓ OK"
        elif n_in_range < n_samples:
            status = f"✗ 仅{n_in_range}点"
        else:
            status = f"⚠️ 采{len(cpp_idx)}"
        
        x_range = f"[{range_obj.min_x:.0f},{range_obj.max_x:.0f}]"
        y_range = f"[{range_obj.min_y:.0f},{range_obj.max_y:.0f}]"
        z_range = f"[{range_obj.min_z:.0f},{range_obj.max_z:.0f}]"
        
        print(f"{label:<20} | {x_range:>16} | {y_range:>16} | {z_range:>12} | {len(cpp_idx):>8} | {status:>10}")
    
    print("\n结论: 推荐用 compute_bbox_python() 自动计算范围，避免手动指定遗漏数据")


def verify_step_by_step(points=None, n_samples=20):
    """逐步骤详细对比"""
    print("\n" + "=" * 70)
    print("逐步骤详细对比")
    print("=" * 70)
    
    if points is None:
        np.random.seed(123)
        points = np.random.randn(100, 3).astype(np.float32)
    
    # Python 计算 SpaceRange
    range_obj = compute_bbox_python(points, 16, 16, 16)
    
    py_idx, py_dists = fps_numpy(points, n_samples)
    cpp_idx = yf.fps(points, n_samples, range_obj)
    cpp_dists = compute_distance_sequence(points, cpp_idx)
    
    n = min(len(py_idx), len(cpp_idx), n_samples)
    
    print(f"{'Step':>5} | {'PyIdx':>6} | {'CppIdx':>6} | {'IdxMatch':>8} | {'PyDist':>10} | {'CppDist':>10} | {'DistDiff':>10}")
    print("-" * 85)
    
    for i in range(n):
        idx_match = "✓" if py_idx[i] == cpp_idx[i] else "✗"
        dist_diff = abs(py_dists[i] - cpp_dists[i]) if i < len(cpp_dists) else float('nan')
        dist_diff_str = f"{dist_diff:.2e}" if dist_diff < 1e6 else "N/A"
        marker = " <--" if dist_diff > 1e-3 else ""
        
        print(f"{i:>5} | {py_idx[i]:>6} | {cpp_idx[i]:>6} | {idx_match:>8} | "
              f"{py_dists[i]:>10.4f} | {cpp_dists[i]:>10.4f} | {dist_diff_str:>10}{marker}")


def benchmark_speed():
    """性能基准测试"""
    print("\n" + "=" * 70)
    print("性能基准测试")
    print("=" * 70)
    
    sizes = [1000, 5000, 10000, 50000]
    sample_ratio = 0.1
    
    print(f"{'Size':>8} | {'Samples':>8} | {'Python(ms)':>12} | {'C++(ms)':>10} | {'Speedup':>8}")
    print("-" * 65)
    
    for n in sizes:
        np.random.seed(42)
        points = np.random.randn(n, 3).astype(np.float32)
        k = int(n * sample_ratio)
        
        # Python
        t0 = time.time()
        fps_numpy(points, k)
        py_time = (time.time() - t0) * 1000
        
        # C++（Python 计算 range）
        range_obj = compute_bbox_python(points, 16, 16, 16)
        t0 = time.time()
        yf.fps(points, k, range_obj)
        cpp_time = (time.time() - t0) * 1000
        
        speedup = py_time / cpp_time if cpp_time > 0 else float('inf')
        
        print(f"{n:>8} | {k:>8} | {py_time:>12.2f} | {cpp_time:>10.2f} | {speedup:>8.1f}x")


def visualize_distance_decay(save_path="figs/fps_decay.png"):
    """可视化距离衰减曲线"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed, skipping visualization")
        return
    
    print("\n生成距离衰减对比图...")
    
    np.random.seed(42)
    points = np.random.randn(5000, 3).astype(np.float32)
    n_samples = 500
    
    # Python 计算 range
    range_obj = compute_bbox_python(points, 16, 16, 16)
    
    py_idx, py_dists = fps_numpy(points, n_samples)
    cpp_idx = yf.fps(points, n_samples, range_obj)
    cpp_dists = compute_distance_sequence(points, cpp_idx)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左图：距离衰减曲线
    ax1 = axes[0]
    steps = range(1, min(len(py_dists), len(cpp_dists)))
    ax1.semilogy(steps, py_dists[1:len(steps) + 1], 'b-', label='Python', alpha=0.7, linewidth=2)
    ax1.semilogy(steps, cpp_dists[1:len(steps) + 1], 'r--', label='C++', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Sample Step', fontsize=11)
    ax1.set_ylabel('Max Distance (log scale)', fontsize=11)
    ax1.legend()
    ax1.set_title('Distance Decay Comparison', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 右图：距离差异
    ax2 = axes[1]
    min_len = min(len(py_dists), len(cpp_dists))
    diffs = np.abs(py_dists[:min_len] - cpp_dists[:min_len])
    ax2.semilogy(range(min_len), diffs, 'g-', alpha=0.7)
    ax2.axhline(y=1e-3, color='r', linestyle='--', label='threshold 1e-3')
    ax2.set_xlabel('Sample Step', fontsize=11)
    ax2.set_ylabel('|Distance Difference|', fontsize=11)
    ax2.set_title(f'Max Diff: {np.max(diffs):.2e}', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  保存到: {save_path}")


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    results = []
    
    # 核心验证
    results.append(("基础正确性", verify_correctness()))
    results.append(("粒度配置", verify_granularity()))
    
    # 演示手动范围的风险
    verify_manual_range()
    
    # 其他测试
    verify_step_by_step()
    benchmark_speed()
    visualize_distance_decay()
    
    # 总结
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:<20}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("-" * 70)
    if all_passed:
        print("✓ 所有核心验证通过！")
        print("=" * 70)
        print("使用方式:")
        print("  1. Python 计算 SpaceRange:")
        print("     range_obj = compute_bbox_python(points, x_blocks, y_blocks, z_blocks)")
        print("  2. 或者手动指定:")
        print("     range_obj = yf.make_range(min_x, max_x, ..., x_blocks, y_blocks, z_blocks)")
        print("  3. 调用 FPS:")
        print("     indices = yf.fps(points, n_samples, range_obj)")
    else:
        print("✗ 部分验证失败")
    print("=" * 70)