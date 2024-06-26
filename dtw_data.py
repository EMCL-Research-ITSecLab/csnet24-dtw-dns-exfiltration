import numpy as np
import polars as pl

from dtw_utils import multidimensional_to_numpy

CONSTANT_CLASS_MALICIOUS = [2]
CONSTANT_CLASS_BENIGN = [1]


def group_cic_data(input_dir, filenames, class_type, interval="1s", length=5):
    X_ent = []
    y_ent = []

    X_packet_size = []
    y_packet_size = []
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

        for frame in x.iter_slices(n_rows=5):
            if (
                frame["entropy"].sum() > 0
                and frame["entropy"].len() > 4
                and frame["entropy"].to_list().count(0) < 3
            ):
                X_ent.append(frame.select(["entropy"]).to_numpy().reshape(-1))
                if class_type == "0":
                    y_ent.append(CONSTANT_CLASS_BENIGN)
                else:
                    y_ent.append(CONSTANT_CLASS_MALICIOUS)

                X_packet_size.append(
                    frame.select(["packet_size"]).to_numpy().reshape(-1)
                )
                if class_type == "0":
                    y_packet_size.append(CONSTANT_CLASS_BENIGN)
                else:
                    y_packet_size.append(CONSTANT_CLASS_MALICIOUS)

    return X_ent, y_ent, X_packet_size, y_packet_size


if __name__ == "__main__":
    X_ent_cicm, y_ent_cicm, X_packet_size_cicm, y_packet_size_cicm = group_cic_data(
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

    X_ent_cicb, y_ent_cicb, X_packet_size_cicb, y_packet_size_cicb = group_cic_data(
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

    X_ent = X_ent_cicb + X_ent_cicm
    y_ent = y_ent_cicb + y_ent_cicm
    X_packet_size = X_packet_size_cicb + X_packet_size_cicm
    y_packet_size = y_packet_size_cicb + y_packet_size_cicm

    np.save(f"data/x_cic_1min_entropy.npy", np.array(X_ent))
    np.save(f"data/y_cic_1min_entropy.npy", np.array(y_ent))

    np.save(f"data/x_cic_1min_packet_size.npy", np.array(X_packet_size))
    np.save(f"data/y_cic_1min_packet_size.npy", np.array(y_packet_size))

    X_ent, y_ent, X_packet_size, y_packet_size = group_cic_data(
        "./data_cic/dns2tcp",
        ["2018-03-23-11-08-11"],
        "1",
    )

    np.save(f"data/x_dns2tcp_1min_entropy.npy", np.array(X_ent))
    np.save(f"data/y_dns2tcp_1min_entropy.npy", np.array(y_ent))

    np.save(f"data/x_dns2tcp_1min_packet_size.npy", np.array(X_packet_size))
    np.save(f"data/y_dns2tcp_1min_packet_size.npy", np.array(y_packet_size))

    X_ent, y_ent, X_packet_size, y_packet_size = group_cic_data(
        "./data_cic/dnscapy",
        ["2018-03-29-19-06-25"],
        "1",
    )

    np.save(f"data/x_dnscapy_1min_entropy.npy", np.array(X_ent))
    np.save(f"data/y_dnscapy_1min_entropy.npy", np.array(y_ent))

    np.save(f"data/x_dnscapy_1min_packet_size.npy", np.array(X_packet_size))
    np.save(f"data/y_dnscapy_1min_packet_size.npy", np.array(y_packet_size))

    X_ent, y_ent, X_packet_size, y_packet_size = group_cic_data(
        "./data_cic/iodine",
        ["2018-03-19-19-06-24"],
        "1",
    )

    np.save(f"data/x_iodine_1min_entropy.npy", np.array(X_ent))
    np.save(f"data/y_iodine_1min_entropy.npy", np.array(y_ent))

    np.save(f"data/x_iodine_1min_packet_size.npy", np.array(X_packet_size))
    np.save(f"data/y_iodine_1min_packet_size.npy", np.array(y_packet_size))

    X_ent, y_ent, X_packet_size, y_packet_size = group_cic_data(
        "./data_cic/plain",
        ["2018-03-19-19-34-33"],
        "1",
    )

    np.save(f"data/x_plain_1min_entropy.npy", np.array(X_ent))
    np.save(f"data/y_plain_1min_entropy.npy", np.array(y_ent))

    np.save(f"data/x_plain_1min_packet_size.npy", np.array(X_packet_size))
    np.save(f"data/y_plain_1min_packet_size.npy", np.array(y_packet_size))

    X_ent, y_ent, X_packet_size, y_packet_size = group_cic_data(
        "./data_cic/tuns",
        ["2018-03-30-09-40-10"],
        "1",
    )

    np.save(f"data/x_tuns_1min_entropy.npy", np.array(X_ent))
    np.save(f"data/y_tuns_1min_entropy.npy", np.array(y_ent))

    np.save(f"data/x_tuns_1min_packet_size.npy", np.array(X_packet_size))
    np.save(f"data/y_tuns_1min_packet_size.npy", np.array(y_packet_size))
