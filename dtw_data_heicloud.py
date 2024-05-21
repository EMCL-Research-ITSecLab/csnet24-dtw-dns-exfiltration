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

    x = pl.read_csv(
        "/home/smachmeier/results_2024-01-15_45d/*.txt",
        separator=" ",
        try_parse_dates=False,
        has_header=False,
    ).with_columns(
        [
            (pl.col("column_1").str.strptime(pl.Datetime).cast(pl.Datetime)),
            (pl.lit(0)).alias("class"),
        ]
    )

    x = x.rename(
        {
            "column_1": "timestamp",
            "column_2": "return_code",
            "column_3": "client_ip",
            "column_4": "dns_server",
            "column_5": "query",
            "column_6": "type",
            "column_7": "answer",
            "column_8": "size",
        }
    )
    x = x.select(["client_ip", "size", "timestamp", "class"])

    x = x.with_columns(
        [
            pl.col(i).rank("dense").cast(pl.Int64).name.suffix("_encoded")
            for i in ["client_ip", "class"]
        ]
    )
    x = x.drop(["client_ip", "class"])

    x: pl.DataFrame = (
        x.sort("timestamp")
        .group_by_dynamic(
            "timestamp",
            every="1m",
            closed="right",
            offset="1d",
            by=["client_ip_encoded", "class_encoded"],
        )
        .agg(pl.col("size").count())
    )
    print(x)

    # min_date = x.select(["timestamp"]).min().item()
    # max_date = x.select(["timestamp"]).max().item()

    # # We generate empty datetime with zero values in a time range of 6h
    # datetimes = x.select(
    #     pl.datetime_range(
    #         min_date.replace(microsecond=0),
    #         max_date.replace(microsecond=0),
    #         "5m",
    #         time_unit="ms",
    #     ).alias("timestamp")
    # )

    # ids = x.select(["client_ip_encoded", "class_encoded"]).unique()

    # # Cross joining all domain
    # all_dates = datetimes.join(ids, how="cross")
    # # Fill with null
    # x = all_dates.join(
    #     x, how="left", on=["client_ip_encoded", "class_encoded", "timestamp"]
    # ).fill_null(0)

    x = (
        x.group_by(["client_ip_encoded", "class_encoded"])
        .agg(pl.col("size"))
        .with_columns([(pl.col("size").list.slice(0, 100))])
    )

    x = x.filter(pl.col("size").list.len() == 100)

    Y = x.select(["class_encoded"])
    x = x.select(["size"])

    x = multidimensional_to_numpy(x["size"])
    np.save("data/x_heicloud_1min_packet_count.npy", x)
    Y = Y.to_numpy()
    np.save("data/y_heicloud_1min_packet_count.npy", Y)
