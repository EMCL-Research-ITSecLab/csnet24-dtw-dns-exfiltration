import math
from itertools import pairwise
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import polars as pl

TS_TYPE = ["univariate", "multivariate"]

HEICLOUD_DATA = [
    "2023-11-30_2023-12-01_sorted_heiCLOUD_DNS_responses",
    "2023-12-01_2023-12-02_sorted_heiCLOUD_DNS_responses",
    "2023-12-02_2023-12-03_sorted_heiCLOUD_DNS_responses",
    "2023-12-03_2023-12-04_sorted_heiCLOUD_DNS_responses",
    "2023-12-04_2023-12-05_sorted_heiCLOUD_DNS_responses",
    "2023-12-05_2023-12-06_sorted_heiCLOUD_DNS_responses",
    "2023-12-06_2023-12-07_sorted_heiCLOUD_DNS_responses",
    "2023-12-07_2023-12-08_sorted_heiCLOUD_DNS_responses",
    "2023-12-08_2023-12-09_sorted_heiCLOUD_DNS_responses",
    "2023-12-09_2023-12-10_sorted_heiCLOUD_DNS_responses",
    "2023-12-10_2023-12-11_sorted_heiCLOUD_DNS_responses",
    "2023-12-11_2023-12-12_sorted_heiCLOUD_DNS_responses",
    "2023-12-12_2023-12-13_sorted_heiCLOUD_DNS_responses",
    "2023-12-13_2023-12-14_sorted_heiCLOUD_DNS_responses",
    "2023-12-14_2023-12-15_sorted_heiCLOUD_DNS_responses",
    "2023-12-15_2023-12-16_sorted_heiCLOUD_DNS_responses",
    "2023-12-16_2023-12-17_sorted_heiCLOUD_DNS_responses",
    "2023-12-17_2023-12-18_sorted_heiCLOUD_DNS_responses",
    "2023-12-18_2023-12-19_sorted_heiCLOUD_DNS_responses",
    "2023-12-19_2023-12-20_sorted_heiCLOUD_DNS_responses",
    "2023-12-20_2023-12-21_sorted_heiCLOUD_DNS_responses",
    "2023-12-21_2023-12-22_sorted_heiCLOUD_DNS_responses",
    "2023-12-22_2023-12-23_sorted_heiCLOUD_DNS_responses",
    "2023-12-23_2023-12-24_sorted_heiCLOUD_DNS_responses",
    "2023-12-24_2023-12-25_sorted_heiCLOUD_DNS_responses",
    "2023-12-25_2023-12-26_sorted_heiCLOUD_DNS_responses",
    "2023-12-26_2023-12-27_sorted_heiCLOUD_DNS_responses",
    "2023-12-27_2023-12-28_sorted_heiCLOUD_DNS_responses",
    "2023-12-28_2023-12-29_sorted_heiCLOUD_DNS_responses",
    "2023-12-29_2023-12-30_sorted_heiCLOUD_DNS_responses",
    "2023-12-30_2023-12-31_sorted_heiCLOUD_DNS_responses",
    "2023-12-31_2024-01-01_sorted_heiCLOUD_DNS_responses",
    "2024-01-01_2024-01-02_sorted_heiCLOUD_DNS_responses",
    "2024-01-02_2024-01-03_sorted_heiCLOUD_DNS_responses",
    "2024-01-03_2024-01-04_sorted_heiCLOUD_DNS_responses",
    "2024-01-04_2024-01-05_sorted_heiCLOUD_DNS_responses",
    "2024-01-05_2024-01-06_sorted_heiCLOUD_DNS_responses",
    "2024-01-06_2024-01-07_sorted_heiCLOUD_DNS_responses",
    "2024-01-07_2024-01-08_sorted_heiCLOUD_DNS_responses",
    "2024-01-08_2024-01-09_sorted_heiCLOUD_DNS_responses",
    "2024-01-09_2024-01-10_sorted_heiCLOUD_DNS_responses",
    "2024-01-10_2024-01-11_sorted_heiCLOUD_DNS_responses",
    "2024-01-11_2024-01-12_sorted_heiCLOUD_DNS_responses",
    "2024-01-12_2024-01-13_sorted_heiCLOUD_DNS_responses",
    "2024-01-13_2024-01-14_sorted_heiCLOUD_DNS_responses",
]

TIME_INTERVAL_CONFIG = [
    {"time_interval": "1s", "minimum_length": 15, "time_interval_name": "15s"},
    {"time_interval": "1s", "minimum_length": 30, "time_interval_name": "30s"},
    {"time_interval": "1s", "minimum_length": 60, "time_interval_name": "1min"},
    {"time_interval": "1s", "minimum_length": 90, "time_interval_name": "1.5min"},
]

DATA_CONFIG = [
    {"name": "cic-malicious", "input_dir": "./dtw_data/cic/attack", "class_type": "1"},
    {"name": "cic-malicious", "input_dir": "./dtw_data/cic/benign", "class_type": "1"},
    {"name": "dns2tcp", "input_dir": "./dtw_data/dns2tcp", "class_type": "1"},
    {"name": "dnscapy", "input_dir": "./dtw_data/dnscapy", "class_type": "1"},
    {"name": "iodine", "input_dir": "./dtw_data/iodine", "class_type": "1"},
    {"name": "plain", "input_dir": "./dtw_data/plain", "class_type": "0"},
    {"name": "normal", "input_dir": "./dtw_data/normal/normal", "class_type": "0"},
    {
        "name": "crossEndPoint-AndIodine-MX",
        "input_dir": "./dtw_data/crossEndPoint/AndIodine-MX",
        "class_type": "1",
    },
    {
        "name": "crossEndPoint-AndIodine-SRV",
        "input_dir": "./dtw_data/crossEndPoint/AndIodine-SRV",
        "class_type": "1",
    },
    {
        "name": "crossEndPoint-AndIodine-TXT",
        "input_dir": "./dtw_data/crossEndPoint/AndIodine-TXT",
        "class_type": "1",
    },
    {
        "name": "crossEndPoint-AndIodine-CNAME",
        "input_dir": "./dtw_data/crossEndPoint",
        "class_type": "1",
    },
    {
        "name": "crossEndPoint-AndIodine-MX",
        "input_dir": "./dtw_data/crossEndPoint",
        "class_type": "1",
    },
    {
        "name": "crossEndPoint-AndIodine-NULL",
        "input_dir": "./dtw_data/crossEndPoint",
        "class_type": "1",
    },
    {
        "name": "crossEndPoint-AndIodine-SRV",
        "input_dir": "./dtw_data/crossEndPoint",
        "class_type": "1",
    },
    {
        "name": "crossEndPoint-AndIodine-TXT",
        "input_dir": "./dtw_data/crossEndPoint",
        "class_type": "1",
    },
    {
        "name": "tunnel-dns2tcp-key",
        "input_dir": "./dtw_data/tunnel/dns2tcp-key",
        "class_type": "1",
    },
    {
        "name": "tunnel-dns2tcp-cname",
        "input_dir": "./dtw_data/tunnel/dnscat2-cname",
        "class_type": "1",
    },
    {
        "name": "tunnel-dns2tcp-mx",
        "input_dir": "./dtw_data/tunnel/dnscat2-mx",
        "class_type": "1",
    },
    {
        "name": "tunnel-dns2tcp-txt",
        "input_dir": "./dtw_data/tunnel/dnscat2-txt",
        "class_type": "1",
    },
    {"name": "tunnel-dns-shell", "input_dir": "./dtw_data/tunnel", "class_type": "1"},
    {"name": "tunnel-dnscat2", "input_dir": "./dtw_data/tunnel", "class_type": "1"},
    {"name": "tunnel-dnspot", "input_dir": "./dtw_data/tunnel", "class_type": "1"},
    {"name": "tunnel-iodine", "input_dir": "./dtw_data/tunnel", "class_type": "1"},
    {"name": "tunnel-tuns", "input_dir": "./dtw_data/tunnel", "class_type": "1"},
    {"name": "wildcard", "input_dir": "./dtw_data/wildcard", "class_type": "1"},
    {
        "name": "unkownTunnel-tcp-over-dns-CNAME",
        "input_dir": "./dtw_data/unkownTunnel/tcp-over-dns-CNAME",
        "class_type": "1",
    },
    {
        "name": "unkownTunnel-tcp-over-dns-TXT",
        "input_dir": "./dtw_data/unkownTunnel/tcp-over-dns-TXT",
        "class_type": "1",
    },
    {
        "name": "unkownTunnel-cobalstrike",
        "input_dir": "./dtw_data/unkownTunnel",
        "class_type": "1",
    },
    {
        "name": "unkownTunnel-cobalstrike",
        "input_dir": "./dtw_data/unkownTunnel",
        "class_type": "1",
    },
    {
        "name": "unkownTunnel-cobalstrike",
        "input_dir": "./dtw_data/unkownTunnel",
        "class_type": "1",
    },
    {
        "name": "unkownTunnel-cobalstrike",
        "input_dir": "./dtw_data/unkownTunnel",
        "class_type": "1",
    },
    {
        "name": "unkownTunnel-cobalstrike",
        "input_dir": "./dtw_data/unkownTunnel",
        "class_type": "1",
    },
    {
        "name": "unkownTunnel-cobalstrike",
        "input_dir": "./dtw_data/unkownTunnel",
        "class_type": "1",
    },
    {"name": "tuns", "input_dir": "./dtw_data/tuns", "class_type": "1"},
]


def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FN += 1

    return TP, FP, TN, FN


def fdr(y_actual, y_pred):
    TP, FP, TN, FN = perf_measure(y_actual, y_pred)
    if (FP + TP) == 0:
        return 0
    return FP / (FP + TP)


def fpr(y_actual, y_pred):
    TP, FP, TN, FN = perf_measure(y_actual, y_pred)
    if (FP + TN) == 0:
        return 0
    return FP / (FP + TN)


def fttar(y_actual, y_pred):
    TP, FP, TN, FN = perf_measure(y_actual, y_pred)
    if (TP) == 0:
        return 0
    return FP / TP


def multidimensional_to_numpy(s):
    dimensions = [1, len(s)]
    while s.dtype == pl.List:
        s = s.explode()
        dimensions.append(len(s))
    dimensions = [p[1] // p[0] for p in pairwise(dimensions)]
    return s.to_numpy().reshape(dimensions)


def shannon_entropy(string):
    "Calculates the Shannon entropy of a string"

    # get probability of chars in string
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]

    # calculate the entropy
    entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob])

    return entropy


def load_dataset(
    time_interval_name,
    ts_type: str = "univariate",
    dt: str = "entropy",
    data=DATA_CONFIG,
):
    x_arr = []
    y_arr = []
    for data_type in data:
        y = np.load(f"dtw_data_npy/y_{data_type['name']}_{time_interval_name}_{dt}.npy")

        if ts_type == "mutlivariate":
            # dt flag not used, because both are loaded anyways
            X = np.stack(
                [
                    np.load(
                        f"dtw_data_npy/x_{data_type['name']}_{time_interval_name}_entropy.npy"
                    ),
                    np.load(
                        f"dtw_data_npy/x_{data_type['name']}_{time_interval_name}_packet_size.npy"
                    ),
                ],
                axis=1,
            )
        else:
            X = np.load(
                f"dtw_data_npy/x_{data_type['name']}_{time_interval_name}_{dt}.npy"
            )
        if X.size != 0:

            X = MinMaxScaler().fit_transform(X)
            y_arr.append(y)
            x_arr.append(X)
        else:
            print(f"WARNING: {data_type['name']} for {time_interval_name} is empty!")
    X = np.concatenate(x_arr)
    y = np.concatenate(y_arr)
    y = y.reshape(-1)

    return X, y
