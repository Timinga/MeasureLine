import argparse
import cv2
import numpy as np
import pyzed.sl as sl
import open3d as o3d  # 用于点云可视化
from MeasureAnything import MeasureAnything
import os
from demo_utils import get_click_coordinates, display_with_overlay, scale_points


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="3D Reconstruction using MeasureAnything with SAM2 segmentation")
    parser.add_argument('--input_svo', type=str, required=True, help='Path to the input .SVO file')
    parser.add_argument('--thin_and_long', action=argparse.BooleanOptionalAction, help='Flag variable that decides whether to skeletonize or use symmetry axis')
    parser.add_argument('--stride', type=int, help='Stride used to calculate line segments')
    args = parser.parse_args()

    directory_name = os.path.split(args.input_svo)[1].split('.')[0]

    # Initialize the ZED camera
    zed = sl.Camera()

    # Set up the ZED camera parameters
    init_params = sl.InitParameters(camera_disable_self_calib=True)
    init_params.set_from_svo_file(args.input_svo)
    init_params.coordinate_units = sl.UNIT.CENTIMETER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_minimum_distance = 0.2

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED camera")
        return

    # Retrieve camera calibration parameters
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    fx, fy = calibration_params.left_cam.fx, calibration_params.left_cam.fy
    cx, cy = calibration_params.left_cam.cx, calibration_params.left_cam.cy
    k1, k2, p1, p2, k3 = calibration_params.left_cam.disto[0:5]

    # Initialize runtime parameters
    runtime_parameters = sl.RuntimeParameters(enable_fill_mode=True)
    RGB = sl.Mat()
    depth_for_display = sl.Mat()

    frame_count = 0
    prompt_data = {'positive_points': [], 'negative_points': [], 'clicked': False}

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            frame_count += 1

            # Retrieve RGB frame and Depth frame
            zed.retrieve_image(RGB, sl.VIEW.LEFT)
            image_ocv = RGB.get_data()
            image_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)

            zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
            image_depth = depth_for_display.get_data()
            image_depth = image_depth[:, :, 0]  # Get depth channel

            # Resize images for visualization
            display_width, display_height = 1600, 900
            resized_image = cv2.resize(image_rgb, (display_width, display_height))
            resized_depth = cv2.resize(image_depth, (display_width, display_height))

            # Display frame with basic instructions
            instructions = ["Press 's' to select frame"]
            display_with_overlay(image_rgb, image_depth, [], [], [], display_dimensions=[display_width, display_height])

            # Wait for key input
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
                    display_with_overlay(resized_image, None, prompt_data['positive_points'],
                                         prompt_data['negative_points'], [])

                    if key == ord('c'):  # Continue once points are collected
                        break

                # Remove mouse callback
                cv2.setMouseCallback("Video Feed", lambda *unused: None)

                # Scale up the prompts to the original image dimensions
                scale_x = image_rgb.shape[1] / display_width
                scale_y = image_rgb.shape[0] / display_height
                positive_prompts = scale_points(prompt_data['positive_points'], scale_x, scale_y)
                negative_prompts = scale_points(prompt_data['negative_points'], scale_x, scale_y)

                # Initialize MeasureAnything object
                object = MeasureAnything(zed=zed, window=30)
                object.detect_mask(image=resized_image, positive_prompts=positive_prompts, negative_prompts=negative_prompts)

                # Process the mask
                object.process_mask()

                # Calculate line segment coordinates and depth
                line_segment_coordinates, depth = object.calculate_line_segment_coordinates_and_depth()

                # Generate point cloud from depth data
                point_cloud = generate_point_cloud_from_depth(depth, resized_image.shape, fx, fy, cx, cy)

                # Visualize the point cloud
                visualize_point_cloud(point_cloud)

                # Save point cloud data to file
                if not os.path.exists(f"./output/{directory_name}/results_frame_{frame_count}"):
                    os.makedirs(f"./output/{directory_name}/results_frame_{frame_count}")

                np.save(f"./output/{directory_name}/results_frame_{frame_count}/point_cloud.npy", point_cloud)

                # Display interactive instructions
                overlay_text = [f"Frame: {frame_count}", "Press 'q' to quit or 'c' to continue"]
                while True:
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    if key == ord('c'):
                        break

    # Close camera and windows
    zed.close()
    cv2.destroyAllWindows()


def generate_point_cloud_from_depth(depth_values, image_shape, fx, fy, cx, cy):
    """
    通过深度值计算三维点云，返回每个像素点的空间坐标。
    """
    height, width = image_shape
    points = []
    for y in range(height):
        for x in range(width):
            Z = depth_values[y, x]
            if Z > 0:
                X = (x - cx) * Z / fx  # fx, cx 是相机内参
                Y = (y - cy) * Z / fy  # fy, cy 是相机内参
                points.append([X, Y, Z])

    return np.array(points)


def visualize_point_cloud(points):
    """
    使用 Open3D 库进行点云的可视化。
    """
    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Convert the numpy array to Open3D point cloud format
    pcd.points = o3d.utility.Vector3dVector(points)

    # Optionally, you can add color to the points (e.g., based on depth)
    colors = np.array([[0.0, 0.0, 1.0] for _ in range(len(points))])  # Blue color for all points
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud", width=800, height=600)


if __name__ == '__main__':
    main()
