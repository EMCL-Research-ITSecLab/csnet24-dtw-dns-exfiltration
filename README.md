# Dynamic-Time-Warping for DNS Exfiltration Detection

Detection of DNS tunneling using Dynamic-Time-Warping.

## DNS Tunneling

- [iodine](dns/iodine/README.md)
- [dnscat2](dns/dnscat2/README.md)
- [dns2tcp](dns/dns2tcp/README.md)

## Data set

In this study, we investigated the effects of `packet_size`, `entropy`, ``. We converted all `*.pcap` files using Tshark. Due to same IP addresses across the data set, we changed them accordingly.

```sh
tshark -r benign_heavy_3.pcap -Y "dns.flags == 0x0100" -T fields -e frame.time -e ip.src -e frame.len
sed -i -- 's/192.168.20.38/1.1.1.3/g' benign_heavy_3.csv
```

- [CIC-Bell-DNS-EXF-2021]()

In addition, we also traced DNS tunnels in our lab environment.

## Verification

In order to validate our results, we implemented a XGBoost Ensemble and validated our performance against a well-established method.