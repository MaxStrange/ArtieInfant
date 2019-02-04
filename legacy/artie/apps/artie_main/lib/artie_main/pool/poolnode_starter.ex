defmodule ArtieMain.PoolnodeStarter do
  @moduledoc """
  This module is a shim layer for starting a remote node
  Pypool process.
  """

  def start_link(node, poolname, poolsupname) do
    name = poolsupname |> Atom.to_string()
    name = name <> "_shim" |> String.to_atom()
    GenServer.start_link(__MODULE__, {node, poolname, poolsupname}, name: name)
  end

  def init({node, poolname, poolsupname}) do
    _pid = Node.spawn_link(node, ArtieMain.PypoolSupervisor, :start_link, [self(), poolname, poolsupname])
    {:ok, [node, poolname, poolsupname]}
  end

  def handle_info({:child, childpid}, [node, poolname, poolsupname]) do
    Printex.prints "Started pool #{inspect poolname} on #{inspect node}", :light_green
    Process.link(childpid)
    Process.monitor(childpid)
    {:noreply, [node, poolname, poolsupname, childpid]}
  end
end
