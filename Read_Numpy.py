import numpy as np
import matplotlib.pyplot as plt
import os  # 添加os模块用于路径操作

file_path = r"C:\Users\HP\Desktop\measure-anything-main\output\2K200test\results_frame_8\diameters.npy"

try:
    # Load and clean data
    raw_data = np.load(file_path)
    data = raw_data[~np.isnan(raw_data)]

    # Calculate dynamic axis range based on actual data
    data_min = np.floor(data.min() * 2) / 2  # Round down to nearest 0.5
    data_max = np.ceil(data.max() * 2) / 2  # Round up to nearest 0.5
    bin_step = 0.3

    # Create bins aligned with actual data range
    bins = np.arange(data_min, data_max + bin_step, bin_step)

    # Create figure
    plt.figure(figsize=(12, 7))

    # Plot histogram
    hist_values, bin_edges, _ = plt.hist(
        data,
        bins=bins,
        edgecolor='black',
        alpha=0.8
    )

    # Configure x-axis to show actual values
    plt.xticks(
        ticks=np.arange(data_min, data_max + 0.5, 0.5),  # 0.5 increment ticks
        labels=[f'{x:.1f}' for x in np.arange(data_min, data_max + 0.5, 0.5)]
    )

    # Configure labels
    plt.xlabel("Diameter Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Diameter Distribution Histogram", fontsize=14)

    # Add value labels on bars
    for i in range(len(hist_values)):
        if hist_values[i] > 0:
            plt.text(
                bin_edges[i] + bin_step / 2,
                hist_values[i] + 0.5,
                str(int(hist_values[i])),
                ha='center'
            )

    # Add grid and styling
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.gca().set_axisbelow(True)  # Grid behind bars

    # Add data statistics
    info_text = (
        f'Data range: {data.min():.2f} - {data.max():.2f}\n'
        f'Total samples: {len(data)}\n'
        f'Mean (2-5): {np.mean(data[(data > 2) & (data < 5)]):.2f}'
        if (data > 2).any() and (data < 5).any()
        else 'No data in 2-5 range'
    )

    plt.text(
        0.95, 0.90,
        info_text,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.9)
    )
    # 在plt.show()前添加保存逻辑
    save_dir = os.path.dirname(file_path)  # 获取原始文件的目录
    save_path = os.path.join(save_dir, "diameter_distribution.png")  # 拼接保存路径
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存图片
    print(f"图表已保存至：{save_path}")

    plt.show()

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"Runtime error: {str(e)}")