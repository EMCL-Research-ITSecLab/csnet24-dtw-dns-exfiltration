import math
import numpy as np
import polars as pl

from utils import multidimensional_to_numpy


def group_heicloud_data(input_dir, filenames, class_type, interval="1s", length=5):
    dfs = []
    for file in filenames:
        df = pl.read_csv(
            f"{input_dir}/{file}.txt",
            separator=" ",
            try_parse_dates=False,
            has_header=False,
        ).with_columns(
            [
                (pl.col("column_1").str.strptime(pl.Datetime).cast(pl.Datetime)),
                (pl.lit(0)).alias("class"),
            ]
        )

        df = df.rename(
            {
                "column_1": "timestamp",
                "column_2": "return_code",
                "column_3": "src_ip",
                "column_4": "dns_server",
                "column_5": "query",
                "column_6": "type",
                "column_7": "answer",
                "column_8": "packet_size",
            }
        )
        df = df.with_columns(
            [
                (
                    pl.col("query").map_elements(
                        lambda x: [
                            float(str(x).count(c)) / len(str(x))
                            for c in dict.fromkeys(list(str(x)))
                        ]
                    )
                ).alias("prob"),
            ]
        )

        t = math.log(2.0)

        df = df.with_columns(
            [
                # - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
                (
                    pl.col("prob")
                    .list.eval(-pl.element() * pl.element().log() / t)
                    .list.sum()
                ).alias(f"entropy"),
            ]
        )
        df = df.drop("prob")

        df: pl.DataFrame = (
            df.sort("timestamp")
            .group_by_dynamic(
                "timestamp", every=interval, closed="right", by=["src_ip", "class"]
            )
            .agg(pl.col("entropy").mean(), pl.col("packet_size").mean())
        )

        min_date = df.select(["timestamp"]).min().item()
        max_date = df.select(["timestamp"]).max().item()

        # We generate empty datetime with zero values in a time range of 6h
        datetimes = df.select(
            pl.datetime_range(
                min_date.replace(microsecond=0),
                max_date.replace(microsecond=0),
                interval,
                time_unit="us",
            ).alias("timestamp")
        )

        ids = df.select(["src_ip", "class"]).unique()

        # Cross joining all domain
        all_dates = datetimes.join(ids, how="cross")
        # Fill with null
        x = all_dates.join(
            df, how="left", on=["src_ip", "class", "timestamp"]
        ).fill_null(0)

        df = x
        
        dfs.append(df[:length])

    return dfs


if __name__ == "__main__":
    data = [
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
    for day in data:
        total = group_heicloud_data(
            "/home/smachmeier/results_2024-01-15_45d/",
            [
                day
            ],
            "0",
        )
        x = pl.concat(total)
        x = x.with_columns(
            [
                pl.col(i).rank("dense").cast(pl.Int64).name.suffix("_encoded")
                for i in ["src_ip", "class"]
            ]
        )
        x = x.drop(["src_ip", "class"])

        for column in ["packet_size", "entropy"]:
            x_column = x.group_by(["src_ip_encoded", "class_encoded"]).agg(pl.col(column))

            Y_column = x_column.select(["class_encoded"])
            x_column = x_column.select([column])

            x_column = multidimensional_to_numpy(x_column[column])
            np.save(f"data/x_{day}_1min_{column}.npy", x_column)
            Y_column = Y_column.to_numpy()
            np.save(f"data/y_{day}_1min_{column}.npy", Y_column)
