import trimesh
import numpy as np
from trimesh.scene.scene import Scene

def load_glb_file(file_path):
    """加载GLB文件并返回合并后的网格对象（保留颜色）"""
    try:
        mesh = trimesh.load(
            file_path,
            maintain_order=True,
            process=False,
            force='mesh'
        )
        print(f"成功加载GLB文件: {file_path}")
        
        if isinstance(mesh, Scene):
            print("检测到GLB场景，合并所有子网格...")
            valid_meshes = [m for m in mesh.geometry.values() if hasattr(m, 'vertices')]
            if not valid_meshes:
                print("场景中无有效网格数据")
                return None
            mesh = trimesh.util.concatenate(valid_meshes)
        
        # 检查模型在世界坐标系中的位置
        model_center = np.mean(mesh.vertices, axis=0)
        print(f"模型中心在世界坐标系中的位置: X={model_center[0]:.4f}, Y={model_center[1]:.4f}, Z={model_center[2]:.4f}")
        
        return mesh
    except Exception as e:
        print(f"加载GLB文件失败: {e}")
        return None

def calculate_model_size(mesh):
    """计算模型的边界框尺寸"""
    if not mesh or not hasattr(mesh, 'vertices'):
        return None
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    print(f"模型大小：宽{size[0]:.4f} | 高{size[1]:.4f} | 深{size[2]:.4f}")
    return size, bounds

def remove_outliers(mesh, z_threshold=3):
    """去除离离群点（保留顶点颜色）"""
    if not mesh or not hasattr(mesh, 'vertices'):
        return None
    
    vertices = mesh.vertices
    mean = np.mean(vertices, axis=0)
    std = np.std(vertices, axis=0)
    z_scores = np.abs((vertices - mean) / std)
    non_outliers = np.all(z_scores < z_threshold, axis=1)
    
    if isinstance(mesh, trimesh.Trimesh):
        valid_faces = np.all(non_outliers[mesh.faces], axis=1)
        new_indices = np.cumsum(non_outliers) - 1
        new_faces = new_indices[mesh.faces[valid_faces]]
        new_vertices = vertices[non_outliers]
        
        new_vertex_colors = None
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            new_vertex_colors = mesh.visual.vertex_colors[non_outliers]
        
        cleaned_mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            vertex_colors=new_vertex_colors
        )
        print(f"离群点处理完成：原始顶点{len(vertices)} → 处理后{len(new_vertices)}")
        return cleaned_mesh
    
    elif isinstance(mesh, trimesh.PointCloud):
        new_vertices = vertices[non_outliers]
        new_colors = mesh.colors[non_outliers] if hasattr(mesh, 'colors') else None
        cleaned_mesh = trimesh.Pointcloud.PointCloud(new_vertices, colors=new_colors)
        return cleaned_mesh
    
    return None

def create_world_coordinate_frame(axis_length=1.0):
    """
    创建世界坐标系（原点固定在(0,0,0)）
    X轴：红色，Y轴：绿色，Z轴：蓝色
    修复颜色参数错误，确保每个实体都有对应的颜色
    """
    # 世界坐标系原点固定在(0,0,0)
    origin = np.array([0, 0, 0])
    
    # 创建单个线段的函数（带颜色）
    def create_colored_line(start, end, color):
        # 创建线段
        line = trimesh.load_path([start, end])
        # 为每个线段实体设置颜色（每个线段一个实体）
        line.colors = [color]  # 关键修复：用列表包裹颜色，确保每个实体有一个颜色
        return line
    
    # X轴（红色）- RGB值范围调整为0-1
    x_axis = create_colored_line(origin, [axis_length, 0, 0], [1.0, 0, 0])
    
    # Y轴（绿色）
    y_axis = create_colored_line(origin, [0, axis_length, 0], [0, 1.0, 0])
    
    # Z轴（蓝色）
    z_axis = create_colored_line(origin, [0, 0, axis_length], [0, 0, 1.0])
    
    # 尝试添加轴标签
    try:
        from trimesh.creation import text
        # X轴标签
        x_text = text("X", font_size=0.1, depth=0.01)
        x_text.apply_translation([axis_length, 0, 0])
        x_text.visual.vertex_colors = [1.0, 0, 0, 1.0]
        
        # Y轴标签
        y_text = text("Y", font_size=0.1, depth=0.01)
        y_text.apply_translation([0, axis_length, 0])
        y_text.visual.vertex_colors = [0, 1.0, 0, 1.0]
        
        # Z轴标签
        z_text = text("Z", font_size=0.1, depth=0.01)
        z_text.apply_translation([0, 0, axis_length])
        z_text.visual.vertex_colors = [0, 0, 1.0, 1.0]
        
        return trimesh.util.concatenate([x_axis, y_axis, z_axis, x_text, y_text, z_text])
    except Exception as e:
        print(f"无法创建轴标签: {e}")
        # 若无法创建文本标签，仅返回坐标轴
        return trimesh.util.concatenate([x_axis, y_axis, z_axis])

def visualize_with_world_coords(mesh, title="World Coordinate Visualization"):
    """可视化模型和世界坐标系"""
    if not mesh or not hasattr(mesh, 'vertices'):
        print("无法可视化：模型数据不完整")
        return
    
    # 计算模型边界框，动态调整世界坐标轴长度
    _, bounds = calculate_model_size(mesh)
    diag_length = np.linalg.norm(bounds[1] - bounds[0])
    axis_length = diag_length * 0.3  # 坐标轴长度为模型对角线的30%
    
    # 创建世界坐标系（原点固定在(0,0,0)）
    world_coords = create_world_coordinate_frame(axis_length=axis_length)
    
    # 创建场景并添加模型和世界坐标系
    scene = Scene()
    scene.add_geometry(mesh, "model")
    scene.add_geometry(world_coords, "world_coordinates")
    
    # 可视化设置
    scene.show(
        title=title,
        flags={'wireframe': False, 'axis': False},  # 禁用默认小坐标轴
        background=[240, 240, 240, 255],
        resolution=[800, 600]
    )

def process_glb_with_world_coords(file_path, z_threshold=3):
    """处理GLB文件并显示世界坐标系"""
    mesh = load_glb_file(file_path)
    if not mesh:
        return None
    
    print("\n=== 可视化原始模型（世界坐标系） ===")
    visualize_with_world_coords(mesh, title="原始模型与世界坐标系")
    
    cleaned_mesh = remove_outliers(mesh, z_threshold)
    if not cleaned_mesh:
        return None
    
    print("\n=== 可视化处理后模型（世界坐标系） ===")
    visualize_with_world_coords(cleaned_mesh, title="处理后模型与世界坐标系")
    
    return cleaned_mesh

if __name__ == "__main__":
    glb_file_path = "/mnt/nas/liuqipeng/workspace/simulation/assets/nailong.glb"
    process_glb_with_world_coords(glb_file_path, z_threshold=3)
    