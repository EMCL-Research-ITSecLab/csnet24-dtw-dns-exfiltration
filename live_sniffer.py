import asyncio
import datetime

from multiprocessing import Queue
import sched

import joblib
import numpy as np
import polars as pl
from scapy.all import *
from scapy.layers.dns import DNS, DNSQR
from scapy.layers.inet import IP
from scapy.layers.inet6 import IPv6

from utils import shannon_entropy

current_packets = []
example_data = []

async def periodic(pq: Queue, current_packets):
    while True:
        packet = pq.get()
        if packet.haslayer(DNSQR) and (
            packet[DNSQR].qtype == 1
            or packet[DNSQR].qtype == 28
            or packet[DNSQR].qtype == 5
            or packet[DNSQR].qtype == 16
            or packet[DNSQR].qtype == 10
        ):
            time = datetime.utcfromtimestamp(float(packet.time))
            query = packet[DNSQR].qname
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
            current_packets.append(p)

            if len(current_packets) > 1:
                df = pl.DataFrame(current_packets)
                df: pl.DataFrame = (
                    df.sort("timestamp")
                    .group_by_dynamic(
                        "timestamp", every="2s", closed="right", by=["src_ip"]
                    )
                    .agg(pl.col("entropy").mean(), pl.col("packet_size").mean())
                )

                min_date = df.select(["timestamp"]).min().item()
                # min_date = min_date.replace(minute=0, hour=0, second=0, microsecond=0)

                max_date = df.select(["timestamp"]).max().item()
                # max_date = min_date.replace(minute=59, hour=23, second=59, microsecond=59)

                # We generate empty datetime with zero values in a time range of 6h
                datetimes = df.select(
                    pl.datetime_range(
                        min_date.replace(microsecond=0),
                        max_date.replace(microsecond=0),
                        "1s",
                        time_unit="us",
                    ).alias("timestamp")
                )

                ids = df.select(["src_ip"]).unique()

                # Cross joining all domain
                all_dates = datetimes.join(ids, how="cross")
                # Fill with null
                x = all_dates.join(
                    df, how="left", on=["src_ip", "timestamp"]
                ).fill_null(0)
                
                x_column = x.group_by(["src_ip"]).agg(pl.col("entropy"))

                unique_ips = x_column.select(["src_ip"]).unique()
                for row in unique_ips.rows(named=True):
                    data = x_column.filter(pl.col("src_ip") == row["src_ip"])
                    list_entropy = data.select("entropy").rows(named=True)[0]["entropy"]
                    if len(list_entropy) > 4:
                        data = np.asarray(list_entropy[:5])
                        model = joblib.load("dtwknn.pickle")
                        print(model.predict(data))
                        example_data.append(data)
                        y = np.full((len(example_data), 1), 2)
                        # np.save(f"data/x_test_1min_entropy.npy", example_data)
                        # np.save(f"data/y_test_1min_entropy.npy", y)
                        current_packets = []

def stop():
    task.cancel()

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

    def __init__(self, que):
        super().__init__()
        self.que = que
        self.channels = []
        self.s = sched.scheduler(time.time, time.sleep)

    def run(self):
        while True:
            msg = self.que.get()
            self.channels.append(msg)
    
    def check(self):
        print(len(self.channels))
        self.s.enter(2, 1, self.check)
        self.s.run()

# helper function for running a target periodically
async def periodic(interval_sec, coro_name, *args, **kwargs):
    # loop forever
    while True:
        # wait an interval
        await asyncio.sleep(interval_sec)
        # await the target
        coro_name.check(*args, **kwargs)


def main():
    que = Queue()
    sniffer = Sniffer(que)
    consumer = Consumer(que)
    sniffer.start()
    consumer.start()
    consumer.check()
    
    # task = asyncio.create_task(periodic(1.0, consumer))

if __name__ == "__main__":
    main()
    # start the event loop
    # asyncio.run(main())
    # que = Queue()
    # sniffer = Sniffer(que)
    # consumer = Consumer(que)
    # sniffer.start()
    # consumer.start()
    # pq = Queue()
    # # Start async sniffer
    # t = AsyncSniffer(iface="eno1", filter="udp and port 53", store=0, prn=pq.put)
    # t.start()

    # # Create event loop for checking
    # loop = asyncio.get_event_loop()
    # loop.call_later(5, stop)


    # try:
    #     loop.run_until_complete(task)
    # except asyncio.CancelledError:
    #     pass
