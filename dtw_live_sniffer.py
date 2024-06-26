import codecs
import datetime
from multiprocessing import Queue

import joblib
import numpy as np
import polars as pl
from scapy.all import *
from scapy.layers.dns import DNS, DNSQR
from scapy.layers.inet import IP
from scapy.layers.inet6 import IPv6

from dtw_utils import shannon_entropy


class Sniffer(Thread):
    """Producer"""

    def __init__(self, que):
        super().__init__()
        self.que = que

    def run(self):
        while True:
            sniff(prn=self.que.put, iface="eno1", filter="udp and port 53")


class Consumer(Thread):
    """Consumer"""

    def __init__(self, que_data: Queue, que_validator: Queue):
        super().__init__()
        self.que = que_data
        self.que_validator = que_validator
        self.data = []

    def run(self):
        while True:
            packet = self.que.get()
            if (
                packet.haslayer(DNSQR)
                and (
                    packet[DNSQR].qtype == 1
                    or packet[DNSQR].qtype == 28
                    or packet[DNSQR].qtype == 5
                    or packet[DNSQR].qtype == 16
                    or packet[DNSQR].qtype == 10
                )
                and packet.dport == 53
            ):
                time = datetime.utcfromtimestamp(float(packet.time))
                query = codecs.decode(packet[DNSQR].qname, encoding="latin-1")
                src_ip, dst_ip = "", ""
                if IP in packet:
                    src_ip = packet[IP].src
                    dst_ip = packet[IP].dst
                elif IPv6 in packet:
                    src_ip = packet[IPv6].src
                    dst_ip = packet[IPv6].dst
                packet_size = len(packet[DNS])
                entropy = shannon_entropy(query)

                # Create dict object
                p = dict()
                p["timestamp"] = time
                p["src_ip"] = src_ip
                p["dst_ip"] = dst_ip
                p["entropy"] = entropy
                p["packet_size"] = packet_size
                p["query"] = query

                self.que_validator.put(p)


class Validator(Thread):
    """Consumer"""

    def __init__(self, que: Queue, model_path: str = "./dtwknn.pickle"):
        super().__init__()
        self.que: Queue = que
        self.data = []
        self.channels = []
        self.model = joblib.load(model_path)

    def run(self):
        while True:
            self.channels.append(self.que.get())
            if len(self.channels) > 1:
                df_total = pl.DataFrame(self.channels)
                df = (
                    df_total.sort("timestamp")
                    .group_by_dynamic(
                        "timestamp", every="2s", closed="right", by=["src_ip", "dst_ip"]
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
                        "1s",
                        time_unit="us",
                    ).alias("timestamp")
                )

                ids = df.select(["src_ip", "dst_ip"]).unique()

                # Cross joining all domain
                all_dates = datetimes.join(ids, how="cross")
                # Fill with null
                x = all_dates.join(
                    df, how="left", on=["src_ip", "dst_ip", "timestamp"]
                ).fill_null(0)

                x_column = x.group_by(["src_ip", "dst_ip"]).agg(pl.col("entropy"))

                unique_ips = x_column.select(["src_ip", "dst_ip"]).unique()

                for row in unique_ips.rows(named=True):
                    data = x_column.filter(pl.col("src_ip") == row["src_ip"])

                    data_entropy = data.select("entropy").rows(named=True)[0]["entropy"]

                    # Time interval 5 sec.
                    if len(data_entropy) > 4:
                        result = self.model.predict(np.asarray(data_entropy[:5]))
                        if result[0] == 2:
                            print(f"Prediction for data: {result}")
                            print(
                                f"Suspicious Data: {df_total.filter(pl.col('src_ip') == row['src_ip']).sort('timestamp')}"
                            )

                        # self.data.append(data)
                        # y = np.full((len(self.data), 1), 1)
                        # np.save(f"data/x_live_1min_entropy.npy", self.data)
                        # np.save(f"data/y_live_1min_entropy.npy", y)

                        # Clear channels
                        self.channels = []


def main():
    que_data = Queue()
    que_validate = Queue()
    sniffer = Sniffer(que_data)
    consumer = Consumer(que_data, que_validate)
    validator = Validator(que_validate)
    sniffer.start()
    consumer.start()
    validator.start()


if __name__ == "__main__":
    main()
