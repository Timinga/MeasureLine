import cv2
import numpy as np

def enhance_image(image):
    """
    对输入的 BGR 图像进行增强处理：
    1. 转换到 LAB 色彩空间，对 L 通道应用自适应直方图均衡（CLAHE）。
    2. 将增强后的 LAB 图像转换回 BGR。
    3. 使用 unsharp masking 方法增强图像锐度。
    """
    # 转换到 LAB 色彩空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 对 L 通道应用 CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # 合并通道并转换回 BGR 色彩空间
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    image_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 使用高斯模糊与加权相加实现 unsharp masking
    gaussian = cv2.GaussianBlur(image_clahe, (9, 9), 10.0)
    image_sharp = cv2.addWeighted(image_clahe, 1.5, gaussian, -0.5, 0)

    return image_sharp

def main():
    # 在这里直接指定图像文件路径（注意修改为你的实际路径）
    image_path = r"C:\Users\HP\Desktop\rgb.png"  # 使用原始字符串避免转义问题

    # 读取原始图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像，请检查文件路径。")
        return

    # 对图像进行增强处理
    enhanced_image = enhance_image(image)

    # 分别显示原始图像和增强后的图像
    cv2.imshow("Original Image", image)
    cv2.imshow("Enhanced Image", enhanced_image)

    # 分别保存图像（可选）
    cv2.imwrite("original.png", image)
    cv2.imwrite("enhanced.png", enhanced_image)

    # 等待用户按键后退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
