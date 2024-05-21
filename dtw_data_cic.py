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
    df_attacks = []
    for file in ["heavy_image", "heavy_audio", "heavy_compressed", "heavy_exe", "heavy_text", "heavy_video"]:   
        df_attack = pl.read_csv(f"/home/smachmeier/Downloads/DNSExif/Attacks/{file}.csv", has_header=False, separator="\t",try_parse_dates=True).with_columns(
            [
                (pl.col("column_1").str.to_datetime("%B %d, %Y %H:%M:%S%.f CET", time_unit="ms")),
                (pl.lit("1")).alias("class")
            ]
        )
        df_attack: pl.DataFrame = df_attack.sort("column_1").group_by_dynamic("column_1", every="1m", closed="right", by=["column_2", "class"]).agg(pl.col("column_3").count())

        df_attacks.append(df_attack[:100])
    df_benigns=[]
    for file in ["benign_heavy_1", "benign_heavy_2", "benign_heavy_3"]:  
        df_benign = pl.read_csv(f"/home/smachmeier/Downloads/DNSExif/Benign/{file}.csv", has_header=False, separator="\t",try_parse_dates=True).with_columns(
            [
                (pl.col("column_1").str.to_datetime("%B %d, %Y %H:%M:%S%.f CET", time_unit="ms")),
                (pl.lit("0")).alias("class")
            ]
        )
        
        df_benign: pl.DataFrame = df_benign.sort("column_1").group_by_dynamic("column_1", every="1m", closed="right", by=["column_2", "class"]).agg(pl.col("column_3").count())

        df_benigns.append(df_benign[:100])
    
    total = df_benigns + df_attacks
        
    df = pl.concat(total)
    
    df = df.rename(
        {
            "column_1" : "timestamp",
            "column_2" : "client",
            "column_3" : "size" 
        }
    )
    
    x = df
    
    print(x)

    x = x.with_columns([
        pl.col(i).rank("dense").cast(pl.Int64).name.suffix("_encoded") for i in ["client", "class"]
    ])
    x = x.drop(["client", "class"])

    x = x.group_by(["client_encoded", "class_encoded"]).agg(pl.col("size"))
    
    print(x)

    Y = x.select(["class_encoded"])
    x = x.select(["size"])

    x = multidimensional_to_numpy(x["size"])
    # np.save("x_cic_h_packet.npy", x)
    Y = Y.to_numpy()
    # np.save("y_cic_h_packet.npy", Y)