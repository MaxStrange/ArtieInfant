# Artie

## How To Use

First, you need to make sure every computer that you will be communicating with
has the same ~/.erlang.cookie.

Second, you must make sure that every computer has each other in their /etc/hosts file.
(On Windows, this is located in C:\Windows\System32\drivers\etc\hosts).

Third, you must open an IEx session on each node with the following command:

```bash
iex --sname <name of this node>
```
You should probably do that in a screen session if you want to leave it running forever.
