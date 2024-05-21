import polars as pl
import numpy as np

from itertools import pairwise

def multidimensional_to_numpy(s):
    dimensions = [1, len(s)]
    while s.dtype == pl.List:
        s = s.explode()
        dimensions.append(len(s))
    dimensions = [p[1] // p[0] for p in pairwise(dimensions)]
    return s.to_numpy().reshape(dimensions)


if __name__ == "__main__":

    df = pl.read_csv("/home/smachmeier/Downloads/dataset.csv", has_header=False).with_columns(
        [(pl.from_epoch("column_3", time_unit="ms"))]
    )
    
    df = df.rename(
        {
            "column_1": "user_ip", 		# user identification (IP address) - anonymized
            "column_2": "domain", 			# top level domail (TLD)
            "column_3": "timestamp", 		# timestamp of the request
            "column_4": "attack", 			# indicator if the request is malicious (True/False) 
            "column_5": "request", 		# request text
            "column_6": "len", 			# length of the request (without TLD) -> length(something.google.com) = 9 (not counting .google.com) 
            "column_7": "subdomains_count", 	# number of subdomains (without TLD)  -> subdomains_count(something.there.google.com) = 2 (not counting .google.com) 
            "column_8": "w_count", 		# number of English words in the request
            "column_9": "w_max",			# lenght of the longest English word in the request
            "column_10": "entropy", 		# entropy of DNS request 
            "column_11": "w_max_ratio",		# longest English word length to request length ratio 
            "column_12": "w_count_ratio",		# number of English words to request lenght ratio 
            "column_13": "digits_ratio",		# percentage of digits in the request
            "column_14": "uppercase_ratio",		# percentage of capital letter in the request

            # The following features are calculated using the current and the previous 9 requests (window size = 10).
            # Requests are grouped by (user_ip, domain) key.

            "column_15": "time_avg", 		# average time between requests
            "column_16": "time_stdev", 		# standard deviation of times between requests
            "column_17": "size_avg", 		# average size (length) of the requests
            "column_18": "size stdev", 		# standard deviation of sizes of requests
            "column_19": "throughput", 		# number of characters in requests transmitted per second
            "column_20": "unique", 			# uniqueness indicator with values in range [0-1] (0 - all requests are equal, 1 - all requests are different)
            "column_21": "entropy_avg",		# average value of entropy
            "column_22": "entropy_stdev", 	
        }
    )

    x = df.select(["user_ip", "size_avg", "timestamp", "attack"])
    

    x = x.with_columns([
        pl.col(i).rank("dense").cast(pl.Int64).name.suffix("_encoded") for i in ["user_ip", "attack"]
    ])
    x = x.drop(["user_ip", "attack"])

    x: pl.DataFrame = x.sort("timestamp").group_by_dynamic("timestamp", every="5m", closed="right", by=["user_ip_encoded", "attack_encoded"]).agg(pl.col("size_avg")).with_columns([
        pl.col("size_avg").list.mean()
    ])
    
    min_date = x.select(["timestamp"]).min().item()
    max_date = x.select(["timestamp"]).max().item()

    # We generate empty datetime with zero values in a time range of 6h
    datetimes = x.select(
        pl.datetime_range(
            min_date.replace(microsecond=0), max_date.replace(microsecond=0), "5m", time_unit="ms"
        ).alias("timestamp")
    )

    ids = x.select(["user_ip_encoded", "attack_encoded"]).unique()

    # Cross joining all domain
    all_dates = datetimes.join(ids, how="cross")
    # Fill with null
    x = all_dates.join(
        x, how="left", on=["user_ip_encoded", "attack_encoded", "timestamp"]
    ).fill_null(0)

    x = x.group_by(["user_ip_encoded", "attack_encoded"]).agg(pl.col("size_avg"))
    print(x)

    Y = x.select(["attack_encoded"])
    x = x.select(["size_avg"])

    x = multidimensional_to_numpy(x["size_avg"])
    np.save("x_data_h_sizeavg.npy", x)
    Y = Y.to_numpy()
    np.save("y_data_h_sizeavg.npy", Y)