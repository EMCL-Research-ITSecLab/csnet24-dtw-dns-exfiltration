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
                "Shannon Entropy of Query"
                if dt_type == "entropy"
                else "Packet Size of Query"
            )
            plt.ylabel("Probability")
            plt.title(f"Distribution of Requests")
            plt.xticks(range(2, 6, 1) if dt_type == "entropy" else range(16, 300, 64))
            plt.xlim((2, 6) if dt_type == "entropy" else (16, 300))
            plt.yticks(np.arange(0, 0.6, step=0.1))
            plt.ylim(0, 0.6)
            plt.legend(["Benign", "Malicious"], title="Types")
            plt.show()
            plt.savefig(
                f"figs/data_distribution_{ti['time_interval_name']}_{dt_type}.pdf", bbox_inches='tight'
            )
            plt.clf()
