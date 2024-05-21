import sys

sys.path.append("../conversion")
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.colors as color
from convert_images import filter_data, apply_calibration

dark = None
gain = None


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Plot summed intensity over a peak squared region according to the angle of rotation in fly-scan mode."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="H5 input master file"
    )
    parser.add_argument(
        "-s",
        "--start_frame",
        type=int,
        action="store",
        help="starting frame in the interval of images measured",
    )
    parser.add_argument(
        "-e",
        "--end_frame",
        type=int,
        action="store",
        help="ending frame in the interval of images measured",
    )
    parser.add_argument(
        "-p_x0",
        "--p_x0",
        type=int,
        action="store",
        help="first x position (column) of the peak position square on wich the sum will be calculated",
    )
    parser.add_argument(
        "-p_x1",
        "--p_x1",
        type=int,
        action="store",
        help="las x position (column) of the peak position square on wich the sum will be calculated",
    )
    parser.add_argument(
        "-p_y0",
        "--p_y0",
        type=int,
        action="store",
        help="first y position (row) of the peak position square on wich the sum will be calculated",
    )
    parser.add_argument(
        "-p_y1",
        "--p_y1",
        type=int,
        action="store",
        help="last y position (row) of the peak position square on wich the sum will be calculated",
    )
    parser.add_argument(
        "-p1",
        "--pedestal1",
        type=str,
        action="store",
        help="path to the pedestal file for module 1",
    )
    parser.add_argument(
        "-p2",
        "--pedestal2",
        type=str,
        action="store",
        help="path to the pedestal file for module 2",
    )
    parser.add_argument(
        "-g1",
        "--gain1",
        type=str,
        action="store",
        help="path to the gain info file for module 1",
    )
    parser.add_argument(
        "-g2",
        "--gain2",
        type=str,
        action="store",
        help="path to the gain info file for module 1",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        action="store",
        help="path to the output file",
    )
    args = parser.parse_args(raw_args)

    global dark, gain

    num_panels: int = 2
    dark_filenames = [args.pedestal1, args.pedestal2]
    gain_filenames = [args.gain1, args.gain2]
    dark = np.ndarray((3, 512 * num_panels, 1024), dtype=np.float32)
    gain = np.ndarray((3, 512 * num_panels, 1024), dtype=np.float64)
    panel_id: int
    for panel_id in range(num_panels):
        gain_file: BinaryIO = open(gain_filenames[panel_id], "rb")
        dark_file: Any = h5py.File(dark_filenames[panel_id], "r")
        gain_mode: int
        for gain_mode in range(3):
            dark[gain_mode, 512 * panel_id : 512 * (panel_id + 1), :] = dark_file[
                "gain%d" % gain_mode
            ][:]
            gain[gain_mode, 512 * panel_id : 512 * (panel_id + 1), :] = np.fromfile(
                gain_file, dtype=np.float64, count=1024 * 512
            ).reshape(512, 1024)
        gain_file.close()
        dark_file.close()

    f = h5py.File(f"{args.input}", "r")
    raw = f["entry/data/data"]

    f.close()
    peak_pos_x = [args.p_x0, args.p_x1]
    peak_pos_y = [args.p_y0, args.p_y1]

    raw = np.array(raw[args.start_frame : args.end_frame])
    n_frames = args.end_frame - args.start_frame

    x = raw.shape[1]
    y = raw.shape[2]
    filt_data = raw.copy()

    for i in range(n_frames):
        skip = filter_data(filt_data[i])
        if skip == 1:
            filt_data[i] = dark[0]

    d = np.zeros((n_frames, y, x), dtype=np.float64)
    for idx, i in enumerate(filt_data):
        d[idx] = apply_calibration(i, dark, gain)

    peak_image = np.zeros(
        (n_frames, peak_pos_y[1] - peak_pos_y[0], peak_pos_x[1] - peak_pos_x[0]),
        dtype=np.float64,
    )
    for idx, i in enumerate(d):
        peak_image[idx] = i[
            peak_pos_y[0] : peak_pos_y[1], peak_pos_x[0] : peak_pos_x[1]
        ]

    total_intensity = np.zeros((n_frames,), dtype=np.float64)

    for idx, i in enumerate(peak_image):
        total_intensity[idx] = np.sum(i, axis=None)

    x = np.arange(0, n_frames * 0.01, 0.01)
    print(total_intensity.shape, x.shape)
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    plt.subplot(111)
    plt.plot(x, total_intensity)
    ax.set_ylabel("Intensity a.u.")
    ax.set_xlabel("2​θ (deg)")
    plt.savefig(f"{args.output}")
    plt.show()


if __name__ == "__main__":
    main()
