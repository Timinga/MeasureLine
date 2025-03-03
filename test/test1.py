import cv2
import numpy as np

# 1. 读取图像 (路径请自行替换)
image_path=r"C:\Users\HP\Desktop\left_0.jpg"
img = cv2.imread(image_path)

# 如果读取失败，img 将为 None，需要检查文件路径是否正确
if img is None:
    raise FileNotFoundError("无法读取图像，请检查文件路径！")

# 2. 转换到 LAB 色彩空间（便于使用 CLAHE 来局部增强亮度对比）
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# 3. 对 L (亮度) 通道应用 CLAHE
#    clipLimit 可以理解为限制对比度增强的强度；
#    tileGridSize 则决定分块大小，越大就越接近全局直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

# 将增强后的 L 通道与原先的 A、B 通道合并
lab_clahe = cv2.merge((l_clahe, a, b))
# 再转换回 BGR 色彩空间
img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# 4. Unsharp Masking (反遮罩锐化)
#    - 先对增强后图像做高斯模糊
#    - 再用 addWeighted 叠加模糊图与原图
blurred = cv2.GaussianBlur(img_clahe, (0, 0), 3)
unsharp = cv2.addWeighted(img_clahe, 1.5, blurred, -0.5, 0)
# 参数含义：
#   1.5: 原图权重
#  -0.5: 模糊图的负权重
#    0 : 亮度偏移

# 5. 使用拉普拉斯算子进一步增强边缘
gray_unsharp = cv2.cvtColor(unsharp, cv2.COLOR_BGR2GRAY)
# 拉普拉斯算子，ksize=3 可以根据需求改为 1、5、7 等
laplacian = cv2.Laplacian(gray_unsharp, cv2.CV_16S, ksize=3)
# 转回 8 位图（0~255）
laplacian_8u = cv2.convertScaleAbs(laplacian)
# 转成 3 通道，便于和原图叠加
laplacian_8u_bgr = cv2.cvtColor(laplacian_8u, cv2.COLOR_GRAY2BGR)

# 把拉普拉斯增强结果叠加回图像
# 这里的 0.3 是拉普拉斯图层的权重，可以适度调高或调低
enhanced = cv2.addWeighted(unsharp, 1.0, laplacian_8u_bgr, 0.3, 0)

# 6. 将最终结果写到文件，图像的宽高尺寸不会发生任何改变
cv2.imwrite("output.jpg", enhanced)

# 如果你想在程序里查看，可以用如下方式打开窗口查看（记得最后销毁窗口）：
cv2.imshow("Original", img)
cv2.imshow("CLAHE", img_clahe)
cv2.imshow("Unsharp", unsharp)
cv2.imshow("Enhanced", enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()