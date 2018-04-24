# Artie


## Octopod

### General Idea

The general way that this library works is this:
1. You start up a bunch of elixir nodes (one for each compute node)
1. You write a python program for each node that uses the pyctopod library
found in this directory's priv/pyctopod folder.
1. Call into the pyctopod library to publish and subscribe to topics.
1. Write a master Elixir app that depends on Octopod and which starts up a Pyctopod
session on each compute node.
1. Your Python code does the rest.

### How To Use

First, you need to make sure every computer that you will be communicating with
has the same ~/.erlang.cookie.

Second, you must make sure that every computer has each other in their /etc/hosts file.
(On Windows, this is located in C:\Windows\System32\drivers\etc\hosts).

Third, you must open an IEx session on each node with the following command:

```bash
iex --sname <name of this node>
```
You should probably do that in a screen session if you want to leave it running forever.

Now you can use this library like this:

```elixir
defmodule DoSomething do
    alias Octopod.Pyctopod

    def start() do
        hostname = nameofnode@nameofcomputer
        {:ok, _pid} = Node.start(hostname)
        Node.spawn_link(hostname, fn -> Pyctopod.start(:pyctotest_consume, nil, nil, [python_path: list_of_places_to_add_to_python_path]) end)
        Process.sleep(3_000)
        Node.spawn_link(hostname, fn -> Pyctopod.start(:pyctotest_pub_one_msg, nil, nil, [python_path: list_of_places_to_add_to_python_path]) end)
        # Modify as needed to start more python processes
    end
end
```

Meanwhile, in Python files that you plan to use with this library:

*pyctotest_pub_one_msg.py*
```python
import pyctopod

def main():
    pyctopod.publish("test_topic", "This is a Test".encode("utf8"), from_id="pyctotest_pub_one_msg")

def register_handler(pid):
    """
    Every file that uses the pyctopod library MUST implement this function
    and should contain this code in it.
    I know it's a little hackish. Oh well.
    """
    pyctopod.register_main(main)
    pyctopod.register_handler(pid)
```

*pyctotest_consume.py*
```python
import pyctopod

def handle_message(from_id, topic, msg):
    msg = msg.decode("utf8") + " FROM PYTHON!"
    print(msg)

def main():
    pyctopod.subscribe(topics=["test_topic"], handlers=[handle_message])

def register_handler(pid):
    pyctopod.register_main(main)
    pyctopod.register_handler(pid)
```

Now run the Elixir test.
