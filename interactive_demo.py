import argparse
import cv2
import numpy as np
import pyzed.sl as sl
import os
import matplotlib.pyplot as plt
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


def plot_diameter_distribution(
        file_path,
        lower_bound=2.0,  # 新增下限参数
        upper_bound=5.0,  # 新增上限参数
):

    # 固定参数设置
    save_name = "diameter_distribution.png"
    figsize = (12, 7)
    bin_step = 0.3
    bar_color = 'skyblue'
    text_color = 'navy'
    dpi = 300

    try:
        # 参数有效性检查
        if lower_bound >= upper_bound:
            raise ValueError("统计范围下限必须小于上限")

        # 加载数据
        raw_data = np.load(file_path)
        data = raw_data[~np.isnan(raw_data)]

        if len(data) == 0:
            raise ValueError("数据文件中所有值均为NaN")

        # 创建画布
        plt.figure(figsize=figsize)

        # 动态计算坐标轴范围
        data_min = np.floor(data.min() * 2) / 2
        data_max = np.ceil(data.max() * 2) / 2
        bins = np.arange(data_min, data_max + bin_step, bin_step)

        # 绘制直方图
        hist_values, bin_edges, _ = plt.hist(
            data,
            bins=bins,
            edgecolor='black',
            alpha=0.8,
            color=bar_color
        )

        # 坐标轴设置
        plt.xticks(
            ticks=np.arange(data_min, data_max + 0.5, 0.5),
            labels=[f'{x:.1f}' for x in np.arange(data_min, data_max + 0.5, 0.5)]
        )
        plt.xlabel("Diameter Value", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Diameter Distribution Histogram", fontsize=14)

        # 添加柱状图数值标签
        for i in range(len(hist_values)):
            if hist_values[i] > 0:
                plt.text(
                    bin_edges[i] + bin_step / 2,
                    hist_values[i] + 0.5,
                    str(int(hist_values[i])),
                    ha='center',
                    color=text_color
                )

        # 添加统计信息（使用新参数）
        in_range_mask = (data > lower_bound) & (data < upper_bound)
        if in_range_mask.any():
            mean_value = np.mean(data[in_range_mask])
            info_text = (
                f'Data range: {data.min():.2f} - {data.max():.2f}\n'
                f'Total samples: {len(data)}\n'
                f'Mean ({lower_bound}-{upper_bound}): {mean_value:.2f}'
            )
        else:
            info_text = (
                f'Data range: {data.min():.2f} - {data.max():.2f}\n'
                f'Total samples: {len(data)}\n'
                f'No data in {lower_bound}-{upper_bound} range'
            )

        plt.text(
            0.95, 0.90,
            info_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.9),
            color=text_color
        )

        # 保存图片
        save_dir = os.path.dirname(file_path)
        save_path = os.path.join(save_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"图表已保存至：{save_path}")
        return data, save_path

    except Exception as e:
        plt.close()  # 确保异常时关闭图形
        raise RuntimeError(f"绘图失败: {str(e)}") from e

def main():
    # Argument parser setup
    LOWER_BOUND=2.0
    UPPER_BOUND=5.0
    parser = argparse.ArgumentParser(description="Interactive demo of Measure Anything using SAM-2 with point prompts")
    parser.add_argument('--input_svo', type=str, required=True, help='Path to the input .SVO file')
    parser.add_argument('--thin_and_long', action=argparse.BooleanOptionalAction, help='Flag variable that decides whether to skeletonize or use symmetry axis')
    parser.add_argument('--stride', type=int, help='Stride used to calculate line segments')
    args = parser.parse_args()

    directory_name = os.path.split(args.input_svo)[1].split('.')[0]

    # Initialize command line inputs
    stride = args.stride if args.stride else 10

    # Create a ZED camera object
    zed = sl.Camera()

    # Initialize the ZED camera, specify depth mode, minimum distance
    init_params = sl.InitParameters(camera_disable_self_calib=True)
    # init_params.camera_resolution = sl.RESOLUTION.HD2K  # Use HD1080 video mode
    # init_params.camera_fps = 15  # Set fps at 30
    init_params.set_from_svo_file(args.input_svo)
    init_params.coordinate_units = sl.UNIT.CENTIMETER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
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
            display_width, display_height = 1080, 720
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
                                 lower_bound=LOWER_BOUND,  # 新增
                                 upper_bound=UPPER_BOUND,
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
                                         lower_bound=LOWER_BOUND,  # 新增
                                         upper_bound=UPPER_BOUND,
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

                # Create directory to save results
                if not os.path.exists(f"./output/{directory_name}/results_frame_{frame_count}"):
                    os.makedirs(f"./output/{directory_name}/results_frame_{frame_count}")

                # Save RGB
                cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/rgb.png", image_rgb)
                # 对选定帧进行图像增强处理
                enhanced_image = enhance_image(image_rgb)
                cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/rgb_enhanced.png", enhanced_image)

                # Initialize MeasureAnything object
                object = MeasureAnything(zed=zed, window=25, stride=stride, thin_and_long=args.thin_and_long,
                                         image_file=None)
                object.detect_mask(image=enhanced_image, positive_prompts=positive_prompts,
                                   negative_prompts=negative_prompts)

                # Process depth frame and save as color
                depth_map_norm = cv2.normalize(image_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                color_depth_map = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)
                color_depth_map[image_depth == 0] = [0, 0, 0]
                cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/depth_map.png", color_depth_map)

                # Process mask and save
                object.process_mask()
                processed_mask = object.processed_binary_mask
                cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/processed_mask.png",
                            object.processed_binary_mask_0_255)

                # Skeletonize, obtain skeleton_coordinates
                if args.thin_and_long:
                    object.skeletonize_and_prune()
                    object.augment_skeleton()

                else:
                    object.build_skeleton_from_symmetry_axis()
                ##################################################################################################################3
                # skeleton_bw = (object.skeleton * 255).astype(np.uint8)
                # #cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/skeleton_bw.png", skeleton_bw)
                #
                # skeleton_overlay_direct = enhanced_image.copy()
                # skeleton_points = np.argwhere(object.skeleton == 1)
                # for (py, px) in skeleton_points:
                #     skeleton_overlay_direct[py, px] = [0, 0, 255]
                # cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/skeleton_overlay_direct.png",
                #             skeleton_overlay_direct)
                #
                # # 也可做 addWeighted()
                # skeleton_col = np.zeros_like(enhanced_image)
                # skeleton_col[..., 2] = skeleton_bw
                # blend = cv2.addWeighted(enhanced_image, 0.7, skeleton_col, 0.3, 0)
                # cv2.imwrite(f"./output/{directory_name}/results_frame_{frame_count}/skeleton_overlay_blend.png", blend)
                ##################################################################################################################################
                # Obtain perpendicular line segment coordinates, respective depth

                object.calculate_perpendicular_slope()
                line_segment_coordinates, depth = object.calculate_line_segment_coordinates_and_depth()

                # Calculate measurements
                # diameters = object.calculate_diameter(line_segment_coordinates, depth)
                # volume, length = object.calculate_volume_and_length(line_segment_coordinates, depth)
                volume, length, line_segment_coordinates, diameters = object.calculate_volume_and_length(
                    line_segment_coordinates,
                    depth,
                    length_threshold=100  # ✅ 新增阈值参数
                )
                # Save results
                np.save(f"./output/{directory_name}/results_frame_{frame_count}/diameters.npy", diameters)
                np.save(f"./output/{directory_name}/results_frame_{frame_count}/volume.npy", volume)
                np.save(f"./output/{directory_name}/results_frame_{frame_count}/length.npy", length)

                diameters_path = f"./output/{directory_name}/results_frame_{frame_count}/diameters.npy"
                try:

                    plot_diameter_distribution(diameters_path)
                except Exception as e:
                    print(f"绘图时发生错误: {str(e)}")


                # Display overlay with segmentation and wait for 'c' to continue
                overlay_text = [f"Frame:{frame_count}", "Press 'c' to continue"]
                while True:
                    display_with_overlay(image_rgb,
                                         None,
                                         [],
                                         [],
                                         line_segment_coordinates,
                                         diameters=diameters,
                                         volume=volume,
                                         length=length,
                                         display_dimensions=[display_width, display_height],
                                         lower_bound=LOWER_BOUND,  # 新增
                                         upper_bound=UPPER_BOUND,
                                         save=False,
                                         save_name="",
                                         mask=processed_mask,
                                         overlay_text=overlay_text)

                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        # Save image when continued
                        display_with_overlay(image_rgb,
                                             None,
                                             [],
                                             [],
                                             line_segment_coordinates,
                                             diameters=diameters,
                                             volume=volume,
                                             length=length,
                                             save=True,
                                             display_dimensions=[display_width, display_height],
                                             lower_bound=LOWER_BOUND,  # 新增
                                             upper_bound=UPPER_BOUND,
                                             save_name=f"./output/{directory_name}/results_frame_{frame_count}/final_result.png",
                                             mask=processed_mask,
                                             overlay_text=overlay_text)

                        break

                # Reset prompt data for the next frame
                prompt_data['positive_points'].clear()
                prompt_data['negative_points'].clear()
                prompt_data['clicked'] = False

    # Close camera and windows
    zed.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
