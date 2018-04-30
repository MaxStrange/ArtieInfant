defmodule Robin.NodeLoop do
  @moduledoc """
  Runs on every node that Robin is started with. Is responsible for starting
  the node function and for passing messages back and forth between Robin and
  it.

  Currently, this module just passes any messages on. It may be useful in the
  future for formatting messages and whatnot.
  """
  use GenServer

  def start_link(robin_pid, func) do
    GenServer.start_link(__MODULE__, {robin_pid, func})
  end


  ################# Callbacks ################# 

  def init({robin_pid, func}) do
    me = self()
    pid = spawn_link(fn -> func.(me) end)
    send(robin_pid, me)
    {:ok, {robin_pid, pid, func}}
  end

  def handle_info(msg, {robin_pid, target_pid, func}) do
    send(target_pid, msg)
    receive do
      result -> send(robin_pid, result)
    end
    {:noreply, {robin_pid, target_pid, func}}
  end
end
