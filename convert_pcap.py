import datetime
import fnmatch
import os

from scapy.layers.dns import DNS, DNSQR
from scapy.layers.inet import IP
from scapy.layers.inet6 import IPv6
from scapy.utils import rdpcap

from utils import shannon_entropy


def convert_pcaps(input_dir, output_dir):
    i = 0
    for root, _, filenames in os.walk(input_dir):
        for filename in fnmatch.filter(filenames, "*.pcap"):
            match = os.path.join(root, filename)
            sub_dir = match.replace(input_dir, "")
            sub_dir = sub_dir.replace(".pcap", "")
            if not os.path.exists(
                os.path.realpath(os.path.dirname(f"{output_dir}/{sub_dir}"))
            ):
                try:
                    os.makedirs(
                        os.path.realpath(os.path.dirname(f"{output_dir}/{sub_dir}"))
                    )
                except:
                    pass
            with open(f"{output_dir}{sub_dir}.csv", "a+") as f:
                f.write(f"timestamp,src_ip,dst_ip,query,entropy,packet_size\n")

            dnsPackets = rdpcap(match)
            for packet in dnsPackets:
                if (
                    packet.haslayer(DNSQR)
                    and (
                        packet[DNSQR].qtype == 1
                        or packet[DNSQR].qtype == 28
                        or packet[DNSQR].qtype == 5
                        or packet[DNSQR].qtype == 16
                    )
                    and packet.dport == 53
                ):
                    time = datetime.datetime.utcfromtimestamp(float(packet.time))
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
                    with open(f"{output_dir}/{sub_dir}.csv", "a+") as f:
                        f.write(
                            f"{time},192.168.0.{i},{dst_ip},{query},{entropy},{packet_size}\n"
                        )
            i += 1


if __name__ == "__main__":
    convert_pcaps("./pcaps", "./data_cic")
