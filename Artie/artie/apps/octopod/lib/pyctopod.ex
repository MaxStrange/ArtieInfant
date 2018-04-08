defmodule Pyctopod do
  @moduledoc """
  This is a high-level API module for the Octopod library. This module
  exposes an API for a particular use-case, which should be the most
  common one.
  """
  use GenServer

  @pypath Application.app_dir(:octopod, "priv/pyctopod") |> to_charlist()
  @opts [{:compressed, 5},
         {:call_timeout, 60_000},
         {:start_timeout, 10_000},
         {:python_path, @pypath},
         {:python, 'python'}
        ]

  # Client API

  @doc """
  Call this first. Starts up a pyctopod server to interface with
  Python.

  ## Examples

    iex> Pyctopod.start()
    {:ok, pypid}

  """
  def start() do
    GenServer.start_link(__MODULE__, [])
  end

  @doc """
  Stops the given pypid.

  ## Examples

    iex> {:ok, pypid} = Pyctopod.start()
    iex> Pyctopod.stop(pypid)
    :ok

  """
  def stop(pypid) do
    Octopod.stop(pypid)
  end

  def push(pid, item) do
    GenServer.cast(pid, {:push, item})
  end

  def pop(pid) do
    GenServer.call(pid, :pop)
  end

  # Server (callbacks)

  def init(_state) do
    Octopod.start_cast(:pyctopod, @opts)
  end

  def handle_info(:work, state) do
    {:noreply, state}
  end

  def handle_call(:pop, _from, [h | t]) do
    {:reply, h, t}
  end

  def handle_call(request, from, state) do
    # Call the default implementation from GenServer
    super(request, from, state)
  end

  def handle_cast({:push, item}, state) do
    {:noreply, [item | state]}
  end

  def handle_cast(request, state) do
    super(request, state)
  end
end
