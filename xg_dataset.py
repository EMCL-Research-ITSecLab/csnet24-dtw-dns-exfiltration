import glob
import logging
import math
from string import ascii_lowercase as alc

import polars as pl


def preprocess(x: pl.DataFrame):
    x = x.with_columns(
        [
            (pl.col("query").str.split(".").alias("labels")),
        ]
    )

    x = x.with_columns(
        [
            (pl.col("labels").list.get(-1).alias("tld")),
        ]
    )

    x = x.with_columns(
        [
            # FQDN
            (pl.col("query")).alias("fqdn"),
        ]
    )

    x = x.with_columns(
        [
            # Second-level domain
            (
                pl.when(pl.col("labels").list.len() > 2)
                .then(pl.col("labels").list.get(-2))
                .otherwise(pl.col("labels").list.get(0))
                .alias("secondleveldomain")
            )
        ]
    )

    x = x.with_columns(
        [
            # Third-level domain
            (
                pl.when(pl.col("labels").list.len() > 2)
                .then(
                    pl.col("labels")
                    .list.slice(0, pl.col("labels").list.len() - 2)
                    .list.join(".")
                )
                .otherwise(pl.lit(""))
                .alias("thirdleveldomain")
            ),
        ]
    )

    return x


def transform(x: pl.DataFrame) -> pl.DataFrame:
    """Transform our dataset with new features

    Args:
        x (pl.DataFrame): dataframe with our features

    Returns:
        pl.DataFrame: preprocessed dataframe
    """
    print("Start data transformation")
    x = x.with_columns(
        [
            (pl.col("query").str.split(".").list.len().alias("label_length")),
            (
                pl.col("query")
                .str.split(".")
                .list.max()
                .str.len_chars()
                .alias("label_max")
            ),
            (
                pl.col("query")
                .str.strip_chars(".")
                .str.len_chars()
                .alias("label_average")
            ),
        ]
    )
    # Get letter frequency
    for i in alc:
        x = x.with_columns(
            [
                (
                    pl.col("query")
                    .str.to_lowercase()
                    .str.count_matches(rf"{i}")
                    .truediv(pl.col("query").str.len_chars())
                ).alias(f"freq_{i}"),
            ]
        )
    for level in ["thirdleveldomain", "secondleveldomain", "fqdn"]:
        x = x.with_columns(
            [
                (
                    pl.when(pl.col(level).str.len_chars().eq(0))
                    .then(pl.lit(0))
                    .otherwise(
                        pl.col(level)
                        .str.len_chars()
                        .truediv(pl.col(level).str.len_chars())
                    )
                ).alias(f"{level}_full_count"),
                (
                    pl.when(pl.col(level).str.len_chars().eq(0))
                    .then(pl.lit(0))
                    .otherwise(
                        pl.col(level)
                        .str.count_matches(r"[a-zA-Z]")
                        .truediv(pl.col(level).str.len_chars())
                    )
                ).alias(f"{level}_alpha_count"),
                (
                    pl.when(pl.col(level).str.len_chars().eq(0))
                    .then(pl.lit(0))
                    .otherwise(
                        pl.col(level)
                        .str.count_matches(r"[0-9]")
                        .truediv(pl.col(level).str.len_chars())
                    )
                ).alias(f"{level}_numeric_count"),
                (
                    pl.when(pl.col(level).str.len_chars().eq(0))
                    .then(pl.lit(0))
                    .otherwise(
                        pl.col(level)
                        .str.count_matches(r"[^\w\s]")
                        .truediv(pl.col(level).str.len_chars())
                    )
                ).alias(f"{level}_special_count"),
            ]
        )

    x = x.with_columns(
        [
            (
                pl.concat_list([f"freq_{i}" for i in alc])
                .list.eval(pl.element().std())
                .list.get(0)
            ).alias(f"freq_std"),
            (
                pl.concat_list([f"freq_{i}" for i in alc])
                .list.eval(pl.element().var())
                .list.get(0)
            ).alias(f"freq_var"),
            (
                pl.concat_list([f"freq_{i}" for i in alc])
                .list.eval(pl.element().median())
                .list.get(0)
            ).alias(f"freq_median"),
            (
                pl.concat_list([f"freq_{i}" for i in alc])
                .list.eval(pl.element().mean())
                .list.get(0)
            ).alias(f"freq_mean"),
        ]
    )

    for level in ["thirdleveldomain", "secondleveldomain", "fqdn"]:
        x = x.with_columns(
            [
                (
                    pl.concat_list(
                        [
                            f"{level}_full_count",
                            f"{level}_alpha_count",
                            f"{level}_numeric_count",
                            f"{level}_special_count",
                        ]
                    )
                    .list.eval(pl.element().std())
                    .list.get(0)
                ).alias(f"{level}_std"),
                (
                    pl.concat_list(
                        [
                            f"{level}_full_count",
                            f"{level}_alpha_count",
                            f"{level}_numeric_count",
                            f"{level}_special_count",
                        ]
                    )
                    .list.eval(pl.element().var())
                    .list.get(0)
                ).alias(f"{level}_var"),
                (
                    pl.concat_list(
                        [
                            f"{level}_full_count",
                            f"{level}_alpha_count",
                            f"{level}_numeric_count",
                            f"{level}_special_count",
                        ]
                    )
                    .list.eval(pl.element().median())
                    .list.get(0)
                ).alias(f"{level}_median"),
                (
                    pl.concat_list(
                        [
                            f"{level}_full_count",
                            f"{level}_alpha_count",
                            f"{level}_numeric_count",
                            f"{level}_special_count",
                        ]
                    )
                    .list.eval(pl.element().mean())
                    .list.get(0)
                ).alias(f"{level}_mean"),
            ]
        )

    print("Start entropy calculation")
    for ent in ["fqdn", "thirdleveldomain", "secondleveldomain"]:
        x = x.with_columns(
            [
                (
                    pl.col(ent).map_elements(
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
                # - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
                (
                    pl.col("prob")
                    .list.eval(-pl.element() * pl.element().log() / t)
                    .list.sum()
                ).alias(f"{ent}_entropy"),
            ]
        )
        x = x.drop("prob")
    print("Finished entropy calculation")

    # Fill NaN
    x = x.fill_nan(0)
    # Drop features not useful anymore
    x = x.drop([
                "query",
                "labels",
                "thirdleveldomain",
                "secondleveldomain",
                "fqdn",
                "tld",
            ])

    print("Finished data transformation")

    return x


def load_data():
    files = glob.glob("/home/smachmeier/Downloads/dgarchive/*.csv")
    for file in files:
        name = file.split("/")[-1].split(".")[0]
        df = pl.read_csv(file, has_header=False, separator=",")
        df = df.rename({"column_1": "query"})
        df = df.select("query")
        df = preprocess(df)
        df = transform(df)
        df = df.with_columns([pl.lit("2").alias("class")])
        df.write_csv(f"dgarchive/data_{name}.csv", separator=",")
        
if __name__ == "__main__":
    load_data()