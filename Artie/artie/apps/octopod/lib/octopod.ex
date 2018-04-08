defmodule Octopod do
  @moduledoc """
  This module is the API for the library.

  The way this library works is this:
  1. Your elixir application starts a python instance that runs whatever python code
  2. That python code will want to send files to other runnin python instances
  3. The python code calls into this library (perhaps through a wrapper lib), 'send(fpath, topic)'
  4. This library finds the file specified and sends it (perhaps with compression) to anyone listening to 'topic'
  5. The listening servers receive the file, deserialize it, and hand it over to the python process that they are running

  """
  use GenServer
  alias Octopod.Export

  # Client API

  @doc """
  Starts the python process as a GenServer and returns the pid. Pass
  this pid into the other functions in this module to use it.

  ## Examples

    iex> {:ok, pid} = Octopod.start()
    iex> is_pid(pid)
    true
 
  """
  def start(pyargs \\ []) do
    start_link(pyargs)
  end

  @doc """
  Starts the python process as a GenServer and returns the pid. Just
  like start/1, but also registers any handler with the process.

  Convenience function for:

  ```elixir
  {:ok, pid} = Octopod.start(pyargs)
  answer = Octopod.call(pid, mod, :register_handler, [self()])
  {answer, pid}
  ```

  This means that `mod` must have a `register_handler()` function.

  ## Examples

    iex> privpath = [:code.priv_dir(:octopod), "test"] |> Path.join() |> to_charlist()
    iex> {:ok, pid} = Octopod.start_cast(:test, [{:cd, privpath}])
    iex> is_pid(pid)
    true

    iex> privpath = [:code.priv_dir(:octopod), "pyctopod"] |> Path.join() |> to_charlist()
    iex> {:ok, pid} = Octopod.start_cast(:pyctopod, [{:cd, privpath}])
    iex> is_pid(pid)
    true

  """
  def start_cast(mod, pyargs \\ []) do
    {:ok, pid} = start_link(pyargs)
    :undefined = Octopod.call(pid, mod, :register_handler, [self()])
    {:ok, pid}
  end

  @doc """
  Call this to terminate a running python process.

  ## Examples

    iex> {:ok, pid} = Octopod.start()
    iex> Octopod.stop(pid)
    :ok

  """
  def stop(pypid) do
    GenServer.stop(pypid, :normal)
  end

  @doc """
  Executes `mod.func(args)` synchronously in the python context.

  ## Examples

    iex> {:ok, pypid} = Octopod.start()
    iex> Octopod.call(pypid, :operator, :add, [2, 3])
    5

  """
  def call(pypid, mod, func, args) do
    GenServer.call(pypid, {mod, func, args}, :infinity)
  end

  @doc """
  Passes `msg` to the module registered with `start_cast`. You must use
  `start_cast/2` to get `pypid` and the module registered with `start_cast/2`
  must have a message handler that can handle the type of message being passed.

  ## Examples

    iex> privpath = [:code.priv_dir(:octopod), "test"] |> Path.join() |> to_charlist()
    iex> {:ok, pid} = Octopod.start_cast(:test, [{:cd, privpath}])
    iex> :ok = Octopod.cast(pid, "hello")
    iex> receive do
    ...>   {:ok, "hello FROM PYTHON!"} -> :ok
    ...>   _ -> :err
    ...> after
    ...>   3_000 -> :err_timeout
    ...> end
    :ok

  """
  def cast(pypid, msg) do
    GenServer.cast(pypid, msg)
  end


  # Helper Functions

  defp start_link(pyargs) do
    GenServer.start_link(__MODULE__, pyargs)
  end


  # Server Callbacks

  def init(pyargs) do
    session = Export.start(pyargs)
    {:ok, session}
  end

  def handle_call({mod, func, args}, _from, session) do
    result = Export.call(session, mod, func, args)
    {:reply, result, session}
  end

  def handle_cast(msg, session) do
    Export.cast(session, msg)
    {:noreply, session}
  end

  def handle_info({:python, message}, session) do
    IO.puts("Received message from python: #{inspect message}")
    {:stop, :normal, session}
  end

  def terminate(_reason, session) do
    Export.stop(session)
    :ok
  end
end
