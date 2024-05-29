from itertools import pairwise

import numpy as np
import polars as pl


def multidimensional_to_numpy(s):
    dimensions = [1, len(s)]
    while s.dtype == pl.List:
        s = s.explode()
        dimensions.append(len(s))
    dimensions = [p[1] // p[0] for p in pairwise(dimensions)]
    return s.to_numpy().reshape(dimensions)


if __name__ == "__main__":
    df_attacks = []
    for file in [
        "heavy_image",
        "heavy_audio",
        "heavy_compressed",
        "heavy_exe",
        "heavy_text",
        "heavy_video",
    ]:
        df_attack = pl.read_csv(
            f"/home/smachmeier/Downloads/DNSExif/Attacks/{file}.csv",
            has_header=False,
            separator="\t",
            try_parse_dates=True,
        ).with_columns(
            [
                (
                    pl.col("column_1").str.to_datetime(
                        "%B %d, %Y %H:%M:%S%.f CET", time_unit="ms"
                    )
                ),
                (pl.lit("1")).alias("class"),
            ]
        )
        df_attack: pl.DataFrame = (
            df_attack.sort("column_1")
            .group_by_dynamic(
                "column_1", every="1m", closed="right", by=["column_2", "class"]
            )
            .agg(pl.col("column_3").count())
        )

        # min_date = df_attack.select(["column_1"]).min().item()
        # min_date = min_date.replace(minute=0, hour=0, second=0, microsecond=0)

        # max_date = df_attack.select(["column_1"]).max().item()
        # max_date = min_date.replace(minute=59, hour=23, second=59, microsecond=59)

        # # We generate empty datetime with zero values in a time range of 6h
        # datetimes = df_attack.select(
        #     pl.datetime_range(
        #         min_date.replace(microsecond=0),
        #         max_date.replace(microsecond=0),
        #         "1m",
        #         time_unit="ms",
        #     ).alias("column_1")
        # )

        # ids = df_attack.select(["column_2", "class"]).unique()

        # # Cross joining all domain
        # all_dates = datetimes.join(ids, how="cross")
        # # Fill with null
        # x = all_dates.join(
        #     df_attack, how="left", on=["column_2", "class", "column_1"]
        # ).fill_null(0)

        df_attacks.append(df_attack[:100])
    df_benigns = []
    for file in ["benign_heavy_1", "benign_heavy_2", "benign_heavy_3"]:
        df_benign = pl.read_csv(
            f"/home/smachmeier/Downloads/DNSExif/Benign/{file}.csv",
            has_header=False,
            separator="\t",
            try_parse_dates=True,
        ).with_columns(
            [
                (
                    pl.col("column_1").str.to_datetime(
                        "%B %d, %Y %H:%M:%S%.f CET", time_unit="ms"
                    )
                ),
                (pl.lit("0")).alias("class"),
            ]
        )

        df_benign: pl.DataFrame = (
            df_benign.sort("column_1")
            .group_by_dynamic(
                "column_1", every="1m", closed="right", by=["column_2", "class"]
            )
            .agg(pl.col("column_3").count())
        )

        # min_date = df_benign.select(["column_1"]).min().item()
        # min_date = min_date.replace(minute=0, hour=0, second=0, microsecond=0)

        # max_date = df_benign.select(["column_1"]).max().item()
        # max_date = min_date.replace(minute=59, hour=23, second=59, microsecond=59)

        # # We generate empty datetime with zero values in a time range of 6h
        # datetimes = df_benign.select(
        #     pl.datetime_range(
        #         min_date.replace(microsecond=0),
        #         max_date.replace(microsecond=0),
        #         "1m",
        #         time_unit="ms",
        #     ).alias("column_1")
        # )

        # ids = df_benign.select(["column_2", "class"]).unique()

        # # Cross joining all domain
        # all_dates = datetimes.join(ids, how="cross")
        # # Fill with null
        # x = all_dates.join(
        #     df_benign, how="left", on=["column_2", "class", "column_1"]
        # ).fill_null(0)

        df_benigns.append(df_benign[:100])

    total = df_benigns + df_attacks

    df = pl.concat(total)

    df = df.rename({"column_1": "timestamp", "column_2": "client", "column_3": "size"})

    x = df

    x = x.with_columns(
        [
            pl.col(i).rank("dense").cast(pl.Int64).name.suffix("_encoded")
            for i in ["client", "class"]
        ]
    )
    x = x.drop(["client", "class"])

    x = x.group_by(["client_encoded", "class_encoded"]).agg(pl.col("size"))

    Y = x.select(["class_encoded"])
    x = x.select(["size"])

    x = multidimensional_to_numpy(x["size"])
    np.save("data/x_cic_1min_packet_count.npy", x)
    Y = Y.to_numpy()
    np.save("data/y_cic_1min_packet_count.npy", Y)
