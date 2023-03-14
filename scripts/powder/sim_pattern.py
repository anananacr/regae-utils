import matplotlib.pyplot as plt
import math
from scipy import constants
import argparse
import yaml


def transfer_matrix(j_sol):
    """
    Transfer matrix construction for electron beam path at REGAE.

    Parameters
    ----------
    j_sol: float
        Sol67 current in amperes (A)

    Returns
    ----------
    m: List[float]
        Transfer matrix
    """
    e_kin = 3.66
    f = e_kin * (1.274 + 1.247 * e_kin) / (0.127 + j_sol**2)
    d_sol = 6.45
    d_sample = 5.5
    d_det = 10.42
    d1 = d_sol - d_sample
    d2 = d_det - d_sol
    m = [[1 - d2 / f, d1 * (1 - d2 / f) + d2], [-1 / f, 1 - d1 / f]]
    return m


def main():

    # argument parser
    parser = argparse.ArgumentParser(
        description="Simulate diffraction pattern peaks positions."
    )
    parser.add_argument(
        "-s", "--target_sample", type=str, action="store", help="sample"
    )
    parser.add_argument(
        "-j", "--j_sol", type=float, action="store", help="j sol 67 in amperes"
    )
    parser.add_argument(
        "-f", "--f_cal", type=float, action="store", help="calibration factor A.U."
    )
    parser.add_argument(
        "-w", "--_lambda", type=float, action="store", help="wavelength in angstroms"
    )
    parser.add_argument(
        "-u", "--unit", type=str, action="store", help="x axis parameter x/q/two_theta"
    )
    args = parser.parse_args()

    stream = open("bib_regae_sample.yaml", "r")
    loader = yaml.Loader(stream)
    bib_sample = loader.get_data()
    for i in bib_sample:
        if i["name"] == args.target_sample:
            target_sample = i

    reflections_list = target_sample["reflections"]
    beam_energy = 5.86 * 1e-13
    px_size = 75 * 1e-6
    # _lambda=1e10*constants.h * constants.c / math.sqrt((beam_energy)**2+(2* beam_energy * constants.electron_mass * (constants.c**2)))
    _lambda = args._lambda

    q_t = []
    two_theta_t = []
    x_t = []
    order = []
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    colors = ["blue"]

    for idx, i in enumerate(reflections_list):
        tm = transfer_matrix(args.j_sol)
        if target_sample["group"] == "cubic":
            a = target_sample["a"]
            q = math.sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2) / a

        elif target_sample["group"] == "tetragonal":
            a = target_sample["a"]
            c = target_sample["c"]
            q = math.sqrt(((i[0] ** 2 + i[1] ** 2) / a**2) + ((i[2] ** 2) / c**2))

        elif target_sample["group"] == "orthorombic":
            a = target_sample["a"]
            b = target_sample["b"]
            c = target_sample["c"]
            q = math.sqrt(
                (i[0] ** 2 / a**2) + (i[1] ** 2 / b**2) + (i[2] ** 2 / c**2)
            )

        elif target_sample["group"] == "monoclinic":
            a = target_sample["a"]
            b = target_sample["b"]
            c = target_sample["c"]
            beta = np.pi * target_sample["beta"] / 180
            q = math.sqrt(
                (i[0] ** 2) / ((a**2) * (np.sin(beta) ** 2))
                + ((i[1] ** 2) / (b**2))
                + (i[2] ** 2) / ((c**2) * (np.sin(beta) ** 2))
                - (2 * i[0] * i[1] * np.cos(beta) / (a * c * (np.sin(beta) ** 2)))
            )

        elif target_sample["group"] == "hexagonal":
            a = target_sample["a"]
            c = target_sample["c"]
            q = math.sqrt(
                ((4 / 3) * (i[0] ** 2 + i[0] * i[1] + i[1] ** 2) / a**2)
                + (i[2] ** 2 / c**2)
            )

        q_t.append(q)
        two_theta = _lambda * q * 180 / np.pi
        two_theta_t.append(two_theta)
        x = tm[0][1] * (_lambda * q) / (px_size * args.f_cal)
        x_t.append(x)
        order.append(100 / (idx + 1))

        if args.unit == "x":
            plt.text(x, 110, f"{i}", rotation="vertical", fontsize=12)
        elif args.unit == "q":
            plt.text(q, 110, f"{i}", rotation="vertical", fontsize=12)
        elif args.unit == "two_theta":
            plt.text(two_theta, 110, f"{i}", rotation="vertical", fontsize=12)

    if args.unit == "x":
        ax.set_xlabel("Peak position (pixel)", fontsize=12)
    elif args.unit == "q":
        x_t = q_t
        ax.set_xlabel(r"q ($\AA^{-1}$)", fontsize=12)
    elif args.unit == "two_theta":
        x_t = two_theta_t
        ax.set_xlabel(r"2$\theta$ (deg)", fontsize=12)

    plt.scatter(x_t, order, marker="o", color=colors[0])
    ax.set_ylabel("Reflection order", fontsize=12)
    ax.set_ylim(0, 150)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.xaxis.set_tick_params(labelsize=10)

    for idx, j in enumerate(x_t):
        y_t = np.arange(0, order[idx], 1)
        x_pos = j * np.ones(len(y_t))
        ax.scatter(x_pos, y_t, color=colors[0], marker="|", alpha=0.4)
    print(x_t)
    ax.grid()

    # plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
