import math
import matplotlib.pyplot as plt
from pyclustering.nnet.som import type_conn
import dtwsom

from utils import TIME_INTERVAL_CONFIG, load_dataset

if __name__ == "__main__":
    for ti in TIME_INTERVAL_CONFIG:
        X, y = load_dataset(ti["time_interval_name"])

        structure = type_conn.grid_four

        som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(X))))
        network = dtwsom.DtwSom(som_x, som_y, structure)
        network.train(X, 20)

        plt.tight_layout()
        network.show_distance_matrix()
        plt.savefig(f"figs/som_{ti['time_interval_name']}.pdf")
