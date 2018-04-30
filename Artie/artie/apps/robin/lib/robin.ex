defmodule Robin do
  @moduledoc """
  Robin is a libray for Round-Robining requests through a queue of nodes.
  This library is very simple - it wraps a GenServer that takes a list of
  Nodes to round robin through and then you call it with call or cast, and
  it forwards the request to the next node in the queue.
  """
  use GenServer

  @doc """
  Initializes the Robin Server with a list of nodes that we will Round Robin through.

  Parameters:
    - nodelist:   A list of nodes to execute loopfunc on
    - loopfunc:   The function that will run each node (must be the same function and
                  must loop forever waiting for whatever messages you plan on sending
                  through Robin.call() and Robin.cast(); it needs to take a pid as an
                  argument and return value to that pid whenever a call is used on it).
  """
  def start_link(nodelist, loopfunc) do
    GenServer.start_link(__MODULE__, {nodelist, loopfunc})
  end

  @doc """
  Sends the given message to the next node in the round robin and waits
  for and then returns a response.

  Parameters:
    - pid:        The pid of the Robin you got from Robin.start_link()
    - msg:        The message to send to the process that is running on the Node.
    - timeout:    Optional timeout value in ms to wait for response.
  """
  def call(pid, msg, timeout \\ 5_000) do
    GenServer.call(pid, msg, timeout)
  end

  @doc """
  Sends the given message to the next node in the round robin and
  immediately returns.

  Parameters:
    - pid:    The pid of the Robin you got from Robin.start_link()
    - msg:    The message to send to the process that is running on the Node.
  """
  def cast(pid, msg) do
    GenServer.cast(pid, msg)
  end


  ################# Callbacks ################# 

  def init({[], _}) do
    {:stop, :no_nodes}
  end
  def init({nodelist, func}) when is_list(nodelist) do
    nodelist |> Enum.each(fn n -> Node.connect(n) end)
    pidlist = nodelist |> Enum.map(fn n -> start_loop_on_node(n, func) end)
    {:ok, {nodelist, pidlist}}
  end
  def init(_) do
    {:stop, :badarg}
  end

  def handle_call(msg, _from, {nodelist, pidlist}) do
    # Get the next node
    [next_node | nodelist] = nodelist
    [next_pid | pidlist] = pidlist

    # Send the next node the message
    send(next_pid, msg)

    # wait forever (this whole function will timeout if nothing ever comes)
    response =
      receive do
        x -> x
      end

    # put node on back of list
    nodelist = nodelist ++ [next_node]
    pidlist = pidlist ++ [next_pid]

    {:reply, response, {nodelist, pidlist}}
  end

  def handle_cast(msg, {nodelist, pidlist}) do
    # Get the next node
    [next_node | nodelist] = nodelist
    [next_pid | pidlist] = pidlist

    # Send msg
    send(next_pid, msg)

    # Put node on back of list
    nodelist = nodelist ++ [next_node]
    pidlist = pidlist ++ [next_pid]

    {:noreply, {nodelist, pidlist}}
  end


  ################# Helpers ################# 

  defp start_loop_on_node(n, func) do
    me = self()
    Node.spawn_link(n, fn -> Robin.NodeLoop.start_link(me, func) end)
    receive do
      loop_pid when is_pid(loop_pid) -> loop_pid
      x -> raise "Couldn't get the pid of loop process. Got #{inspect x} instead."
      after 5_000 -> "Couldn't get the pid of the loop process. Timed out."
    end
  end
end
