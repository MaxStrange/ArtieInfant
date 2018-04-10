defmodule Pyctopod do
  @moduledoc """
  This is a high-level API module for the Octopod library. This module
  exposes an API for a particular use-case, which should be the most
  common one.
  """
  use GenServer
  alias Octopod.Export

  @pypath Application.app_dir(:octopod, "priv/pyctopod") |> to_charlist()
  @testpath Application.app_dir(:octopod, "priv/test") |> to_charlist()
  @opts [{:cd, @testpath},
         {:compressed, 5},
         {:call_timeout, :infinity},
         {:start_timeout, 10_000},
         {:python_path, @pypath},
        ]

  # Client API

  @doc """
  Call this first. Starts up a pyctopod server to interface with
  Python.

  Call the target python module. That module must define a init_pyctopod() function
  that takes no arguments, which calls pyctopod.register_handler().

  ## Examples

    iex> {:ok, pypid} = Pyctopod.start(:pyctotest, self())
    iex> is_pid(pypid)
    true

  """
  def start(mod, msgbox_pid \\ nil, eavesdropper_pid \\ nil) do
    # Start up the publisher-to-consumer bridge
    # self() is wrong - it needs to be updated to pyctopid
    {:ok, pub_to_con_bridge} = PubConBridge.start(self(), self(), eavesdropper_pid)

    # If we are testing, we may take the messages ourselves
    msgbox_pid = if (msgbox_pid == nil), do: pub_to_con_bridge, else: msgbox_pid

    {:ok, pid} = GenServer.start_link(__MODULE__, [mod, msgbox_pid])

    # Alert the bridge to the python process pid
    send pub_to_con_bridge, {:pyctopid, pid}

    # Wait a moment for everything to settle, then tell pyctopid we are ready
    Process.sleep(2_000)
    Octopod.cast(pid, {:ok, :go})
    {:ok, pid}
  end

  @doc """
  Stops the given pypid.

  ## Examples

    iex> {:ok, pypid} = Pyctopod.start(:pyctotest, self())
    iex> Pyctopod.stop(pypid)
    :ok

  """
  def stop(pypid) do
    Octopod.stop(pypid)
  end

  @doc """
  Writes {from, topic, msg} to the given pypid.
  """
  def write_to_python(pypid, from, topic, msg) do
    Octopod.cast(pypid, {from, topic, msg})
  end

  # Server (callbacks)

  def init([module, msgbox_pid]) do
    PubSub.start_link()
    Octopod.start_cast(module, @opts, msgbox_pid)
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
    Octopod.stop(session)
    :ok
  end
end
