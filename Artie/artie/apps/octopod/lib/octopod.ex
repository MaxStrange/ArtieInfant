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
  Call this to terminate a running python process.

  ## Examples

    iex> {:ok, pid} = Octopod.start()
    iex> Octopod.stop(pid)
    :ok

  """
  def stop(pypid) do
    GenServer.stop(pypid, :normal)
  end


  # TODO: Remove these functions ####################
  def cast_count(count) do
    {:ok, pid} = start_link([])
    GenServer.cast(pid, {:count, count})
  end

  def call_count(count) do
    {:ok, pid} = start_link([])
    GenServer.call(pid, {:count, count}, :infinity)
  end
  ###################################################


  # Helper Functions

  defp start_link(pyargs) do
    GenServer.start_link(__MODULE__, pyargs)
  end


  # Server Callbacks

  def init(pyargs) do
    session = Export.start(pyargs)
    #TODO: # REMOVE # Export.call(session, :test, :register_handler, [self()])
    {:ok, session}
  end

  def handle_call({:count, count}, _from, session) do
    result = Export.call(session, :test, :long_counter, [count])
    {:reply, result, session}
  end

  def handle_cast({:count, count}, session) do
    Export.cast(session, count)
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
