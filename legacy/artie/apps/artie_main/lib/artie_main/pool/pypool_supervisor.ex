defmodule ArtieMain.PypoolSupervisor do
  @moduledoc """
  This is a Genserver that is responsible for starting the Pypool on the right node.
  """
  use GenServer

  def start_link(parent_pid, poolname, myname) do
    GenServer.start_link(__MODULE__, {parent_pid, poolname}, name: myname)
  end

  ## Callbacks ##

  def init({parent_pid, poolname}) do
    {:ok, pid} = Pypool.start_link(poolname)
    send(parent_pid, {:child, self()})
    {:ok, {parent_pid, poolname, pid}}
  end
end
