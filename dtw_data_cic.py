import numpy as np
import polars as pl

from utils import multidimensional_to_numpy


def group_cic_data(input_dir, filenames, class_type, interval="1s", length=5):
    dfs = []
    for file in filenames:
        df = pl.read_csv(
            f"{input_dir}/{file}.csv",
            has_header=True,
            separator=",",
            try_parse_dates=True,
        ).with_columns(
            [
                (pl.lit(class_type)).alias("class"),
            ]
        )
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
    df_attacks = group_cic_data(
        "./data_cic/cic/attack",
        [
            "heavy_image",
            "heavy_audio",
            "heavy_compressed",
            "heavy_exe",
            "heavy_text",
            "heavy_video",
            "light_audio",
            "light_compressed",
            "light_exe",
            "light_image",
            "light_text",
            "light_video",
        ],
        "1",
    )
    df_tunnel_dns2tcp = group_cic_data(
        "./data_cic/dns2tcp",
        [
            "2018-03-23-11-08-11"
        ],
        "1",
    )
    df_tunnel_dnscapy = group_cic_data(
        "./data_cic/dnscapy",
        [
            "2018-03-29-19-06-25"
        ],
        "1",
    )
    df_tunnel_iodine = group_cic_data(
        "./data_cic/iodine",
        [
            "2018-03-19-19-06-24"
        ],
        "1",
    )
    df_tunnel_plain = group_cic_data(
        "./data_cic/plain",
        [
            "2018-03-19-19-34-33"
        ],
        "1",
    )
    df_tunnel_tuns = group_cic_data(
        "./data_cic/plain",
        [
            "2018-03-30-09-40-10"
        ],
        "1",
    )
    df_benigns = group_cic_data(
        "./data_cic/cic/benign",
        [
            "benign_heavy_1",
            "benign_heavy_2",
            "benign_heavy_3",
            "benign_light_1",
            "benign_light_2",
        ],
        "0",
    )

    total = df_benigns + df_attacks + df_tunnel_dns2tcp + df_tunnel_dnscapy + df_tunnel_iodine + df_tunnel_plain + df_tunnel_tuns

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
        np.save(f"data/x_cic_1min_{column}.npy", x_column)
        Y_column = Y_column.to_numpy()
        np.save(f"data/y_cic_1min_{column}.npy", Y_column)
