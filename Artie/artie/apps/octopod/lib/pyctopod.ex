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

  Parameters:
    - mod:              The python module; if it is not in priv/pyctopod, it needs to
                        be in the python_path. You can add the module to the python_path
                        by sending in opts.
    - msgbox_pid:       Used for testing and should probably be deprecated in favor of
                        eavesdropper_pid.
    - eavesdropper_pid: A testing device for listening in on any messages published.
    - opts:             A keyword list that may include:
                        compressed: An integer value between 1 and 10 to specify how
                        much compression to attempt on messages. Defaults to 5.
                        call_timeout: A timeout (or :infinity - the default) for calling
                        python functions.
                        start_timeout: A timeout (or :infinity) for starting python
                        processes. Defaults to 10_000 (10 seconds).
                        python_path: A list of strings to include in the python path.
  """
  def start(mod, msgbox_pid \\ nil, eavesdropper_pid \\ nil, opts \\ nil) do
    # Start up the publisher-to-consumer bridge
    # self() is wrong - it needs to be updated to pyctopid
    {:ok, pub_to_con_bridge} = PubConBridge.start(self(), self(), eavesdropper_pid)

    # If we are testing, we may take the messages ourselves
    msgbox_pid = if (msgbox_pid == nil), do: pub_to_con_bridge, else: msgbox_pid

    opts = update_default_opts(opts)
    {:ok, pid} = GenServer.start_link(__MODULE__, [mod, msgbox_pid, opts])

    # Alert the bridge to the python process pid
    send pub_to_con_bridge, {:pyctopid, pid}

    # Wait a moment for everything to settle, then tell pyctopid we are ready
    Process.sleep(2_000)
    Octopod.cast(pid, {:ok, :go})
    {:ok, pid}
  end

  # Updates @opts to include any new opts passed in by the user.
  defp update_default_opts(nil) do
    @opts
  end
  defp update_default_opts(opts) do
    if Keyword.has_value?(opts, :python_path) do
      pypath = Keyword.get_values(@opts, :python_path)
      pypath = [pypath | Keyword.get_values(opts, :python_path)]
      opts = Keyword.put(opts, :python_path, pypath)
    end
    # Other than :python_path, the values should just get overridden
    Keyword.merge(@opts, opts)
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

  def init([module, msgbox_pid, opts]) do
    PubSub.start_link()
    Octopod.start_cast(module, opts, msgbox_pid)
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
