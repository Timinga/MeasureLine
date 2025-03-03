
#python svo_recording.py --output_svo_file "C:\Users\HP\Desktop\3DProgram\recordings\watercup.svo2"
#python svo_recording.py --output_svo_file "C:\Users\HP\Desktop\measure-anything-main\svo\watercup.svo2"
import sys
import pyzed.sl as sl
from signal import signal, SIGINT
import argparse
import os


cam = sl.Camera()
# 初始化 ZED 摄像头
init = sl.InitParameters()
# init.camera_resolution = sl.RESOLUTION.HD2K
# init.camera_fps = 15


# Handler to deal with CTRL+C properly
def handler(signal_received, frame):
    cam.disable_recording()
    cam.close()
    sys.exit(0)


signal(SIGINT, handler)


def main():
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NONE  # Set configuration parameters for the ZED
    #init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.async_image_retrieval = False;  # This parameter can be used to record SVO in camera FPS even if the grab loop is running at a lower FPS (due to compute for ex.)

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open", status, "Exit program.")
        exit(1)

    recording_param = sl.RecordingParameters(opt.output_svo_file,
                                             sl.SVO_COMPRESSION_MODE.H264)  # Enable recording with the filename specified in argument
    err = cam.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Recording ZED : ", err)
        exit(1)

    runtime = sl.RuntimeParameters()
    print("SVO is Recording, use Ctrl-C to stop.")  # Start recording SVO, stop with Ctrl-C command
    frames_recorded = 0

    while True:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:  # Check that a new image is successfully acquired
            frames_recorded += 1
            print("Frame count: " + str(frames_recorded), end="\r")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_svo_file', type=str, help='Path to the SVO file that will be written', required=True)
    opt = parser.parse_args()
    if not opt.output_svo_file.endswith(".svo") and not opt.output_svo_file.endswith(".svo2"):
        print("--output_svo_file parameter should be a .svo file but is not : ", opt.output_svo_file, "Exit program.")
        exit()
    main()