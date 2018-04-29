defmodule PyctoWrapper do
  @moduledoc """
  This module serves as a wrapper for the Pyctopod module
  and allows us to start new Python processes on different
  nodes in the cluster, rather than locally.
  """
  use GenServer

  @doc """
  Starts up the Python process in 'mod' on the node given
  by 'name'. The node must have already been connected to us.

  Parameters:
    - name:       The node name (using shortnames) of the node
                  that we want to spawn our python process on.
                  This node must have already been connected to
                  us.
    - mod:        The python module's name as an atom without the
                  .py extension. This must be in the python path
                  of the OS on the remote node. To set the python path
                  before spawning the process, use 'opts'.
    - opts:       The same as Pyctopod.start()'s 'opt's argument.
  """
  def start_link(name, mod, opts) do
    pid = spawn_on_node(name, mod, opts)
    GenServer.start_link(__MODULE__, {name, mod, opts, pid})
  end


  ############## Callbacks ############## 

  def init({name, mod, opts, pid}) do
    {:ok, {name, mod, opts, pid}}
  end


  ############## Private Functions ############## 

  defp spawn_on_node(name, mod, opts) do
    name |> Node.spawn_link(fn -> Pyctopod.start(mod, nil, self(), opts) end)
  end
end
