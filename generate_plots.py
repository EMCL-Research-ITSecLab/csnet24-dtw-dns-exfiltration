import numpy as np
from utils import TIME_INTERVAL_CONFIG, load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    for dt_type in ["entropy", "packet_size"]:
        for ti in TIME_INTERVAL_CONFIG:
            X, y = load_dataset(ti["time_interval_name"], dt=dt_type)

            y1 = y[np.where(y == 1)[0]]
            X1 = X[np.where(y == 1)[0]]
            X2 = X[np.where(y == 2)[0]]

            # Organize Data - from question
            X1 = X1.reshape(-1)
            X1 = X1[X1 > 0]

            X2 = X2.reshape(-1)
            X2 = X2[X2 > 0]

            sns.histplot(X1, bins=30, kde=True, stat="probability")
            sns.histplot(X2, bins=30, kde=True, stat="probability")
            plt.xlabel(
                "Shannon Entropy of Query" if dt_type == "entropy" else "Packet Size"
            )
            plt.ylabel("Probability")
            plt.title(f"Distribution of Requests for {ti['time_interval_name']}")
            plt.xticks(range(0, 8, 1) if dt_type == "entropy" else range(0, 255, 32))
            plt.show()
