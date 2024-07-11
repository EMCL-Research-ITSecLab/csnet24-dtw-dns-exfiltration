# dns2tcp

To start the image in interactive mode (required):

```
docker run --rm -it --privileged -p 53:53/udp -e DOMAIN_NAME="dnscat2.test.de" --name dnscat2 csnet24/dnscat2
```