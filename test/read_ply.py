'''
第五步: 读取点云
输入：ply文件
输出：可视化结果
'''

import open3d as o3d

output_file = "./point_cloud/pointcloud1.ply"
pcd = o3d.io.read_point_cloud(output_file)
o3d.visualization.draw_geometries([pcd])
