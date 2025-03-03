
---
# Demo #
## Interactive Demo ##
The interactive demo requires `.svo` files collected using a ZED stereo camera. Example `.svo` files can be found [here](https://drive.google.com/drive/folders/1Q6). Run the demo and follow onscreen instructions.
```bash
python interactive_demo.py --input_svo path/to/svo/file.svo --stride 10 --thin_and_long
```
- `--thin_and_long` is a flag variable that decides the skeleton construction method. Toggling this flag will construct the skeleton based on skeletonization (recommended for rod-like geometries).
- `--stride (int)` is an optional parameter that determines the distance between consecutive measurements. The default value is 10.
- Red line indicate valid measurements.
- Blue line segments indicate invalid measurements, due to unavailable depth data.
- The calculated stem diameters are available as a numpy file in `./output/{svo_file_name}/{frame}/diameters.npy` ordered from the bottommost to the topmost line measurements.

<p align="center">
<img src="figures/canola.gif" alt="GIF 1" width="98%">
</p>

