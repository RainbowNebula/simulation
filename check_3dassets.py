import trimesh
import numpy as np
import open3d as o3d

def load_glb_file(file_path):
    """加载GLB文件并返回合并后的网格对象"""
    try:
        mesh = trimesh.load(file_path)
        print(f"成功加载GLB文件: {file_path}")
        
        # 如果是场景对象，合并所有网格
        if isinstance(mesh, trimesh.Scene):
            print("检测到场景对象，合并所有网格...")
            meshes = list(mesh.geometry.values())
            mesh = trimesh.util.concatenate(meshes)
            
        return mesh
    except Exception as e:
        print(f"加载GLB文件失败: {e}")
        return None

def calculate_model_size(mesh):
    """计算并返回模型的大小（边界框尺寸）"""
    if not mesh:
        return None
    
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    print(f"模型大小 (宽x高x深): {size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f}")
    return size

def remove_outliers(mesh, z_threshold=3):
    """使用Z-score方法去除离群点"""
    if not mesh or not hasattr(mesh, 'vertices'):
        return None
    
    vertices = mesh.vertices
    mean = np.mean(vertices, axis=0)
    std = np.std(vertices, axis=0)
    z_scores = np.abs((vertices - mean) / std)
    non_outliers = np.all(z_scores < z_threshold, axis=1)
    
    if isinstance(mesh, trimesh.Trimesh):
        faces = mesh.faces
        valid_faces = np.all(non_outliers[faces], axis=1)
        new_indices = np.cumsum(non_outliers) - 1
        new_faces = new_indices[faces[valid_faces]]
        new_vertices = vertices[non_outliers]
        cleaned_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        print(f"已去除离群点: 原始顶点数 {len(vertices)}, 处理后顶点数 {len(new_vertices)}")
        return cleaned_mesh
    else:
        cleaned_vertices = vertices[non_outliers]
        cleaned_mesh = trimesh.PointCloud(cleaned_vertices)
        print(f"已去除离群点: 原始顶点数 {len(vertices)}, 处理后顶点数 {len(cleaned_vertices)}")
        return cleaned_mesh

def trimesh_to_o3d_mesh(trimesh_mesh):
    """将trimesh网格转换为Open3D网格"""
    if isinstance(trimesh_mesh, trimesh.Trimesh):
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
        o3d_mesh.compute_vertex_normals()  # 计算法向量用于光照
        return o3d_mesh
    elif isinstance(trimesh_mesh, trimesh.PointCloud):
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
        return o3d_pcd
    return None

def visualize_with_o3d(mesh, title="3D Model Visualization"):
    """使用Open3D可视化模型和坐标轴"""
    if not mesh or not hasattr(mesh, 'vertices'):
        print("无法可视化：模型数据不完整")
        return
    
    # 转换为Open3D格式
    o3d_mesh = trimesh_to_o3d_mesh(mesh)
    if not o3d_mesh:
        print("无法转换为Open3D格式")
        return
    
    # 创建坐标系（原点，轴长为模型对角线的1/5）
    bounds = mesh.bounds
    diag_length = np.linalg.norm(bounds[1] - bounds[0])
    axis_size = diag_length / 5
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size, origin=np.mean(bounds, axis=0)  # 坐标轴放在模型中心
    )
    
    # 设置可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=800, height=600)
    
    # 添加几何对象
    vis.add_geometry(o3d_mesh)
    vis.add_geometry(coordinate_frame)
    
    # 设置渲染选项（可选）
    opt = vis.get_render_option()
    opt.background_color = [0.9, 0.9, 0.9]  # 浅灰色背景
    opt.mesh_show_back_face = True  # 显示背面
    if isinstance(o3d_mesh, o3d.geometry.PointCloud):
        opt.point_size = 2  # 点云大小
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

def process_glb_file(file_path, z_threshold=3):
    """处理GLB文件的主函数"""
    # 加载模型
    mesh = load_glb_file(file_path)
    if not mesh:
        return
    
    # 检查顶点数据
    if not hasattr(mesh, 'vertices'):
        print("错误：模型没有顶点数据")
        return
    
    # 计算模型大小
    calculate_model_size(mesh)
    
    # 可视化原始模型
    visualize_with_o3d(mesh, "原始模型与坐标轴")
    
    # 去除离群点
    cleaned_mesh = remove_outliers(mesh, z_threshold)
    if not cleaned_mesh:
        return
    
    # 计算处理后模型大小
    print("处理后的模型大小:")
    calculate_model_size(cleaned_mesh)
    
    # 可视化处理后的模型
    visualize_with_o3d(cleaned_mesh, "去除离群点后的模型与坐标轴")
    
    return cleaned_mesh

if __name__ == "__main__":
    # 替换为你的GLB文件路径
    glb_file_path = "/home/haichao/workspace/real2sim/simulation/assets/blue_box.glb"
    
    # 处理GLB文件
    process_glb_file(glb_file_path, z_threshold=3)