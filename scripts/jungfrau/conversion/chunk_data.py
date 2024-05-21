import h5py
import argparse
import numpy as np


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Chop big HDF5 file in smaller HDF5 files."
    )
    parser.add_argument(
        "-i", "--input", type=str, action="store", help="hdf5 input image"
    )

    parser.add_argument(
        "-s", "--start_frame", type=int, action="store", help="starting file index"
    )
    parser.add_argument(
        "-e", "--end_frame", type=int, action="store", help="ending file index"
    )
    parser.add_argument(
        "-f",
        "--frames",
        default=None,
        type=int,
        action="store",
        help="Number of frames to be save in the output file.",
    )

    parser.add_argument(
        "-o", "--output", type=str, action="store", help="hdf5 output path"
    )
    args = parser.parse_args(raw_args)

    n_frames = args.frames
    index = np.arange(args.start_frame, args.end_frame, n_frames)

    for idx, i in enumerate(index):
        f = h5py.File(f"{args.input}", "r")

        size = len(f["data"])

        try:
            data = np.array(f["data"][i : i + n_frames], dtype=np.int32)
        except OSError:
            print("skipped", i)
            continue

        f.close()

        g = h5py.File(f"{args.output}_{idx}.h5", "w")
        g.create_dataset("data", data=data)
        g.close()


if __name__ == "__main__":
    main()
