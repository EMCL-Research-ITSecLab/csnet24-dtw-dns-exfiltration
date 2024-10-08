# Dynamic-Time-Warping for DNS Exfiltration Detection

Detection of DNS tunneling using Dynamic-Time-Warping.

## Getting Started

Convert PCAPs to `.csv` file format:

```sh
python dtw_convert_pcap.py
```

Next, convert CSV files to `.npy` to train our kNN classifier:

```sh
python dtw_data.py
python dtw_data_heicloud.py # not publicly available
```

Next, train classifier (our skip this step to use the provided one)

```sh
python dtwknn.py
```

## DNS Tunneling

- [iodine](dns/iodine/README.md)
- [dnscat2](dns/dnscat2/README.md)
- [dns2tcp](dns/dns2tcp/README.md)
- [dnspot](dns/dnspot/README.md)

## Data set

In this study, we investigated the effects of `packet_size`, `entropy`, ``. We converted all `*.pcap` files using Tshark. Due to same IP addresses across the data set, we changed them accordingly.

```sh
tshark -r benign_heavy_3.pcap -Y "dns.flags == 0x0100" -T fields -e frame.time -e ip.src -e frame.len
sed -i -- 's/192.168.20.38/1.1.1.3/g' benign_heavy_3.csv
```

- [CIC-Bell-DNS-EXF-2021]()


In addition, we also traced DNS tunnels in our lab environment.

Check [heiBOX](https://heibox.uni-heidelberg.de/d/c4654c4d1e054c938ebe/) for the PCAPs files.

## Verification

In order to validate our results, we implemented a XGBoost Ensemble and validated our performance against a well-established method.

## Problems

### skTime and Tensorflow

Set `XLA_FLAGS` for cuda path:
```sh
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
```

Remove warning:

```sh
for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done
```
