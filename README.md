# Dynamic-Time-Warping for DNS Exfiltration Detection


## Data set

### CIC-Bell-DNS-EXF-2021

We converted all `*.pcap` files using Tshark. Due to same IP addresses across the data set, we changed them accordingly.

```sh
tshark -r benign_heavy_3.pcap -Y "dns.flags == 0x0100" -T fields -e frame.time -e ip.src -e frame.len
sed -i -- 's/192.168.20.38/1.1.1.3/g' benign_heavy_3.csv
```