import argparse
import cv2
import numpy as np
import pyzed.sl as sl
import os
import open3d as o3d
from MeasureAnything import MeasureAnything
from demo_utils import get_click_coordinates, display_with_overlay, scale_points

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

    # 合并通道并转换回 BGR
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    image_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 使用 unsharp masking 进行锐化
    gaussian = cv2.GaussianBlur(image_clahe, (9, 9), 10.0)
    image_sharp = cv2.addWeighted(image_clahe, 1.5, gaussian, -0.5, 0)

    return image_sharp


def create_point_cloud(depth_image, mask, zed, color_image=None, output_scale=1.0):
    """
    根据深度图、mask 及可选的 RGB 图像生成带颜色的点云。
    参数：
      - depth_image: 原始深度图（单位为米）
      - mask: 二值 mask，非零区域为目标区域
      - zed: 已打开的 ZED 相机对象，用于获取内参
      - color_image: 原始 RGB 图像（BGR格式），用于提取颜色；若为 None 则使用深度映射颜色
      - output_scale: 输出点云坐标缩放系数（此处默认为1，因为深度单位已是米）
    返回：
      - Open3D 的点云对象
    """
    # 找到 mask 内非零像素的索引
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        print("未在 mask 内找到任何点")
        return None

    # 提取对应的深度值（单位为米）
    depths = depth_image[ys, xs].astype(np.float32)

    # 获取相机内参：左目摄像头的参数
    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    fx = calib.fx
    fy = calib.fy
    cx = calib.cx
    cy = calib.cy

    # 使用 output_scale 转换深度单位（此处默认已为米，无需转换）
    depths_scaled = depths * output_scale

    # 反投影： pixel(u,v) -> (X, Y, Z)
    X = (xs - cx) * depths_scaled / fx
    Y = (ys - cy) * depths_scaled / fy
    Z = depths_scaled

    points = np.stack((X, Y, Z), axis=-1)

    # 创建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 根据是否提供原始 RGB 图像为点云添加颜色
    if color_image is not None:
        # 假设 color_image 与 depth_image 尺寸一致
        if color_image.shape[2] == 3:
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        else:
            color_image_rgb = color_image
        colors = color_image_rgb[ys, xs, :].astype(np.float32) / 255.0
    else:
        # 使用深度值映射的颜色
        colors = cv2.applyColorMap(cv2.normalize(depths, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                   cv2.COLORMAP_JET)
        colors = colors[:, ::-1]  # 转换 BGR -> RGB
        colors = colors.astype(np.float32) / 255.0

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Interactive demo of Measure Anything using SAM-2 with point prompts")
    parser.add_argument('--input_svo', type=str, required=True, help='Path to the input .SVO file')
    parser.add_argument('--thin_and_long', action=argparse.BooleanOptionalAction, help='Flag variable that decides whether to skeletonize or use symmetry axis')
    parser.add_argument('--stride', type=int, help='Stride used to calculate line segments')
    args = parser.parse_args()

    directory_name = os.path.split(args.input_svo)[1].split('.')[0]

    # Create a ZED camera object
    zed = sl.Camera()

    # Initialize the ZED camera, specify depth mode, minimum distance
    init_params = sl.InitParameters(camera_disable_self_calib=True)
    # init_params.camera_resolution = sl.RESOLUTION.HD2K  # Use HD1080 video mode
    # init_params.camera_fps = 15  # Set fps at 30
    init_params.set_from_svo_file(args.input_svo)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_minimum_distance = 0.2
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED camera")
        return

    # Enable fill mode
    runtime_parameters = sl.RuntimeParameters(enable_fill_mode=True)

    RGB = sl.Mat()
    frame_count = 0
    prompt_data = {'positive_points': [], 'negative_points': [], 'clicked': False}

    # Main loop to extract frames and display video
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            frame_count += 1

            # Retrieve the RGB frame
            zed.retrieve_image(RGB, sl.VIEW.LEFT)
            image_ocv = RGB.get_data()
            image_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)

            # Retrieve the depth frame
            depth_for_display = sl.Mat()
            zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
            image_depth = depth_for_display.get_data()
            image_depth = image_depth[:, :, 0]

            # Resize the image for display
            display_width, display_height = 1600, 900
            resized_image = cv2.resize(image_rgb, (display_width, display_height))
            resized_depth = cv2.resize(image_depth, (display_width, display_height))

            # Calculate scale factors for x and y
            scale_x = image_rgb.shape[1] / display_width
            scale_y = image_rgb.shape[0] / display_height

            # Display frame with basic instructions
            instructions = ["Press 's' to select frame"]
            display_with_overlay(image_rgb,
                                 image_depth,
                                 [],
                                 [],
                                 [],
                                 display_dimensions=[display_width, display_height],
                                 diameters=None,
                                 save=False,
                                 save_name="",
                                 mask=None,
                                 overlay_text=instructions)

            # Wait for key inputqc
            key = cv2.waitKey(1)
            if key == ord('q'):  # Quit the loop
                break
            elif key == ord('s'):  # Stop on 's' to select points

                # Set mouse callback for collecting points
                cv2.setMouseCallback("Video Feed", get_click_coordinates, param=prompt_data)

                # Wait for user to select points and press 'c' to continue
                while True:
                    key = cv2.waitKey(1)

                    # Detailed instructions
                    detailed_instructions = [
                        f"Frame: {frame_count}",
                        "'Left-click' to add positive point",
                        "'Ctrl + Left-click' to add negative point",
                        "Press 'c' to continue"
                    ]

                    # Display with current positive and negative points
                    display_with_overlay(resized_image,
                                         None,
                                         prompt_data['positive_points'],
                                         prompt_data['negative_points'],
                                         [],
                                         display_dimensions=[display_width, display_height],
                                         diameters=None,
                                         volume=None,
                                         length=None,
                                         save=False,
                                         save_name="",
                                         mask=None,
                                         overlay_text=detailed_instructions)

                    if key == ord('c'):  # Continue once points are collected
                        break

                # Remove mouse callback
                cv2.setMouseCallback("Video Feed", lambda *unused: None)

                # Scale up the prompts to the original image dimensions
                positive_prompts = scale_points(prompt_data['positive_points'], scale_x, scale_y)
                negative_prompts = scale_points(prompt_data['negative_points'], scale_x, scale_y) if prompt_data[
                    'negative_points'] else None

                # 定义并创建输出目录
                output_dir = f"./output/{directory_name}/results_frame_{frame_count}"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Save RGB
                cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/rgb.png", image_rgb)
                # 对选定帧进行图像增强处理
                enhanced_image = enhance_image(image_rgb)
                cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/rgb_enhanced.png", enhanced_image)

                measurer = MeasureAnything(zed=zed, window=25, stride=args.stride, thin_and_long=args.thin_and_long,
                                           image_file=None)
                measurer.detect_mask(image=enhanced_image, positive_prompts=positive_prompts,
                                     negative_prompts=negative_prompts)

                depth_map_norm = cv2.normalize(image_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                color_depth_map = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)
                color_depth_map[image_depth == 0] = [0, 0, 0]
                cv2.imwrite(os.path.join(output_dir, "depth_map.png"), color_depth_map)

                measurer.process_mask()
                processed_mask = measurer.processed_binary_mask
                cv2.imwrite(os.path.join(output_dir, "processed_mask.png"),
                            measurer.processed_binary_mask_0_255)

                # 确保 mask 与深度图尺寸一致
                if processed_mask.shape[:2] != image_depth.shape:
                    print("警告：mask 与深度图尺寸不匹配，将进行尺寸调整")
                    processed_mask_resized = cv2.resize(processed_mask, (image_depth.shape[1], image_depth.shape[0]),
                                                        interpolation=cv2.INTER_NEAREST)
                else:
                    processed_mask_resized = processed_mask

                # 生成带颜色的点云：这里传入 image_rgb 作为颜色信息
                pcd = create_point_cloud(image_depth, processed_mask_resized, zed, color_image=image_rgb,
                                         output_scale=1.0)
                if pcd is not None:
                    o3d.io.write_point_cloud(os.path.join(output_dir, "point_cloud.ply"), pcd)
                    o3d.visualization.draw_geometries([pcd])
                else:
                    print("未生成点云。")


                # # Reset prompt data for the next frame
                # prompt_data['positive_points'].clear()
                # prompt_data['negative_points'].clear()
                # prompt_data['clicked'] = False

    # Close camera and windows
    zed.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
