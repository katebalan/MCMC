import matplotlib.pyplot as plt

folderPath1 = "tmp/exp1/"
folderPath2 = "tmp/exp2/"


def set_grid_to_plot():
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='grey', linestyle='-')
    plt.grid(b=True, which='minor', color='grey', linestyle='--')
