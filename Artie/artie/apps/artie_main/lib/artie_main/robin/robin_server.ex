defmodule ArtieMain.Robin.Server do
  @moduledoc """
  This is the GenServer that maintains the list of Robin
  instances and the mapping of each function to each Robin.
  """
  use GenServer

  defmodule RobinMetaInfo do
    @moduledoc """
    This is just a struct to hold metadata about Robins.

    Members:
      - func:   The function that this Robin is assigned (note that all Robins in here are
                actually assigned the same underlying loop function, but we maintain
                an assigned function for each of them anyway)
      - nodes:  The list of nodes that this Robin is responsible for
      - pid:    The pid of this Robin
    """
    defstruct [:func, :nodes, :pid]
  end

  @doc """
  Arguments:
    function_node_map:  A map of functions to node names.
  """
  def start_link(function_node_map) do
    GenServer.start_link(__MODULE__, function_node_map, name: __MODULE__)
  end

  def init(function_node_map) do
    # Start each Robin node with this function
    loopfunc = &ArtieMain.Robin.Helper.loop_func/1
    robins =
      function_node_map
      |> Enum.map(fn func, nodes -> {func, nodes, Robin.start_link(nodes, loopfunc)} end)
      |> Enum.map(fn {func, nodes, {:ok, pid}} -> %RobinMetaInfo{func: func, nodes: nodes, pid: pid} end)
    {:ok, {function_node_map, robins}}
  end

  def handle_call({:lookup, func}, _from, {_function_node_map, robins}) do
    result = robins |> Enum.find(nil, fn r -> r.func == func end)
    if result == nil do
      raise "Could not find a Robin responsible for #{inspect func}"
    end
    {:reply, result}
  end
  def handle_call({:execute, pid, m, f, a}, _from, {_function_node_map, _robins}) do
    msg = {:apply_func, m, f, a}
    result = Robin.call(pid, msg)
    {:reply, result}
  end
end
