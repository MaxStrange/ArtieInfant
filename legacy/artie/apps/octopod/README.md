# Octopod

This library attempts to make it easy to send large files between running python
processes on different nodes in a cluster.

## Usage

The idea behind this library is that Python does most of the heavy lifting, but all
the coordination (publisher-subscription, remote process initialization, supervision)
is done by Elixir.

### Python

Create a module like normal, however, it must have the following boilerplate in it:

```python
def main():
    """
    This function will be called after the python process has been spun up
    by the Elixir app. Code in if __name__ == "__main__" blocks will NOT run.
    """
    # Here we just subscribe to a topic, but we can do anything here
    pyctopod.subscribe(topics=["test_topic"], handlers=[handle_message])

def register_handler(pid):
    """
    This function gets called by Elixir to pass control to Python.
    It should look exactly like this.
    """
    pyctopod.register_main(main)
    pyctopod.register_handler(pid)
```

### Elixir

```elixir
    hostname = :"foo@localhost"
    {:ok, _pid} = Node.start(hostname)
    mypid = self()
    _pypid0 = Node.spawn_link(hostname, fn -> Pyctopod.start(:pyctotest_consume, nil, mypid) end)
    Process.sleep(3_000)  # Spawning is asynchronous, so we may publish before we have subscribed
    _pypid1 = Node.spawn_link(hostname, fn -> Pyctopod.start(:pyctotest_pub_one_msg) end)

    assert_receive({:pyctotest_consume, :test, "This is a Test FROM PYTHON!"}, 30_000)
```


## Installation

```elixir
def deps do
  [
    {:octopod, in_umbrella: true}
  ]
end
```

*This section is likely not needed*
*NOTE* Installing erlport is not hard, but is not as simple as just sticking it in
a mix file as a dependency. To finish the installation of this package, you need to
look at the deps/erlport/priv/python2 and python3 directories - and do this:

1. Figure out where you install your pip packages - we'll call this <path_to_python>
1. Copy everything from deps/erlport/priv/python2 into <path_to_python>/Lib/site-packages/erlport
1. Copy everything from deps/erlport/priv/python3 into <path_to_python>/Lib/site-packages/erlport

There is a pip-installable erlport package, but it is currently broken. Someone should really get around to just wrapping the python2 and python3 stuff from erlport's erlang package into a pip package and putting it up on pypi.

