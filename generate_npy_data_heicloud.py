import math

import numpy as np
import polars as pl

from utils import HEICLOUD_DATA, TIME_INTERVAL_CONFIG

CONSTANT_CLASS = [1]


def group_heicloud_data(input_dir, filenames, class_type, interval="1s", length=5):
    X_ent = []
    y_ent = []

    X_packet_size = []
    y_packet_size = []
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
        # Iterate over all unique IDs
        unique_id = df.select(["src_ip"]).unique()
        for row in unique_id.rows(named=True):

            # Run detector
            x = df.filter((pl.col("src_ip") == row["src_ip"]))

            x = x.with_columns(
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

            x = x.with_columns(
                [
                    (
                        pl.col("prob")
                        .list.eval(-pl.element() * pl.element().log() / t)
                        .list.sum()
                    ).alias(f"entropy"),
                ]
            )
            x = x.drop("prob")

            x: pl.DataFrame = (
                x.sort("timestamp")
                .group_by_dynamic(
                    "timestamp", every=interval, closed="right", by=["src_ip", "class"]
                )
                .agg(
                    pl.col("entropy").mean(),
                    pl.col("packet_size").str.strip_chars("b").cast(pl.Int16).mean(),
                )
            )

            min_date = x.select(["timestamp"]).min().item()
            # min_date = min_date.replace(minute=0, hour=0, second=0, microsecond=0)

            max_date = x.select(["timestamp"]).max().item()
            # max_date = min_date.replace(minute=59, hour=23, second=59, microsecond=59)

            # We generate empty datetime with zero values in a time range of 6h
            datetimes = x.select(
                pl.datetime_range(
                    min_date.replace(microsecond=0),
                    max_date.replace(microsecond=0),
                    interval,
                    time_unit="us",
                ).alias("timestamp")
            )

            ids = x.select(["src_ip", "class"]).unique()

            # Cross joining all domain
            all_dates = datetimes.join(ids, how="cross")
            # Fill with null
            x = all_dates.join(
                x, how="left", on=["src_ip", "class", "timestamp"]
            ).fill_null(0)

            for frame in x.iter_slices(n_rows=5):
                if (
                    frame["entropy"].sum() > 0
                    and frame["entropy"].len() > length - 1
                    and frame["entropy"].to_list().count(0) < (length * 0.6)
                ):
                    X_ent.append(frame.select(["entropy"]).to_numpy().reshape(-1))
                    y_ent.append(CONSTANT_CLASS)

                    X_packet_size.append(
                        frame.select(["packet_size"]).to_numpy().reshape(-1)
                    )
                    y_packet_size.append(CONSTANT_CLASS)

    return X_ent, y_ent, X_packet_size, y_packet_size


if __name__ == "__main__":
    for ti in TIME_INTERVAL_CONFIG:
        for day in HEICLOUD_DATA:
            print(f"Start converting: {day} for {ti['time_interval']}")
            X_ent, y_ent, X_packet_size, y_packet_size = group_heicloud_data(
                "/home/smachmeier/results_2024-01-15_45d/",
                [day],
                "0",
                interval=ti["time_interval"],
                length=ti["minimum_length"]
            )
            
            print(f"Finished converting: {day} for {ti['time_interval']}")
            
            np.save(f"dtw_data_npy/x_{day}_{ti['time_interval_name']}_entropy.npy", np.array(X_ent))
            np.save(f"dtw_data_npy/y_{day}_{ti['time_interval_name']}_entropy.npy", np.array(y_ent))

            np.save(f"dtw_data_npy/x_{day}_{ti['time_interval_name']}_packet_size.npy", np.array(X_packet_size))
            np.save(f"dtw_data_npy/y_{day}_{ti['time_interval_name']}_packet_size.npy", np.array(y_packet_size))
            
            print(f"Done: {day} for {ti['time_interval']}")
