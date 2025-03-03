import numpy as np

# 指定文件路径
file_path = r"C:\Users\HP\Desktop\measure-anything-main\output\2K500\results_frame_16\diameters.npy"


# 读取npy文件
try:
    data = np.load(file_path)
    print("文件内容:")
    print(data)
except Exception as e:
    print(f"读取文件时出错: {e}")