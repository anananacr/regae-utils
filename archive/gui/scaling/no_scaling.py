import h5py
import subprocess as sub
import argparse


def main(raw_args=None):

    # argument parser
    parser = argparse.ArgumentParser(description="Function that bypass scaling step.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        help="path to the H5 data master file",
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", help="path to the output H5 file"
    )

    args = parser.parse_args(raw_args)
    cmd = f"cp {args.input} {args.output}.h5"
    sub.call(cmd, shell=True)


if __name__ == "__main__":
    main()
