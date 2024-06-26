import math
from itertools import pairwise

import polars as pl

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
