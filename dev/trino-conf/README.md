### Local Trino Cluster with Self-Signed Cert
 
In order to test Trino with user@password you need a local cluster with Basic Auth enabled. 

## First thing, we need a self signed cert.
```
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout dev/trino-conf/etc/trino-key.pem -out dev/trino-conf/etc/trino-cert.pem -config dev/trino-conf/cert.conf
cat dev/trino-conf/etc/trino-cert.pem dev/trino-conf/etc/trino-key.pem > dev/trino-conf/etc/trino-combined.pem
```

## Include https settings in config.properties

Include the following lines in dev/trino-conf/etc/config.properties

```
http-server.https.enabled=true
http-server.https.keystore.path=/etc/trino/trino-combined.pem
http-server.https.keystore.key=<key>
http-server.authentication.type=PASSWORD
```

## Create a password.db file

```
touch dev/trino-conf/etc/password.db
htpasswd -B -C 10 dev/trino-conf/etc/password.db test
```