# Octopod

This library attempts to make it easy to send large files between running python
processes on different nodes in a cluster.


## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `octopod` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:octopod, "~> 0.1.0"}
  ]
end
```

*NOTE* Installing erlport is not hard, but is not as simple as just sticking it in
a mix file as a dependency. To finish the installation of this package, you need to
look at the deps/erlport/priv/python2 and python3 directories - and do this:

1. Figure out where you install your pip packages - we'll call this <path_to_python>
1. Copy everything from deps/erlport/priv/python2 into <path_to_python>/Lib/site-packages/erlport
1. Copy everything from deps/erlport/priv/python3 into <path_to_python>/Lib/site-packages/erlport

There is a pip-installable erlport package, but it is currently broken. Someone should really get around to just wrapping the python2 and python3 stuff from erlport's erlang package into a pip package and putting it up on pypi.

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at [https://hexdocs.pm/octopod](https://hexdocs.pm/octopod).

