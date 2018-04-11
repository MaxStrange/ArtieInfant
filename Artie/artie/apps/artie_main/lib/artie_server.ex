defmodule ArtieServer do
  @moduledoc """
  This is the main logic for Artie's distributed infrastructure.
  We simply provide a GenServer to start up all the nodes and then
  spin forever.
  """
  @hostname :"master@localhost"
  use GenServer

  @doc """
  Starts the application by going through the list of nodes found
  in the configuration file and starting its associated Python process
  for each one.
  """
  def start_link() do
    GenServer.start_link(__MODULE__, [])
  end


  ############## Callbacks ############## 

  def init([]) do
    # Start us up in distributed mode, using shortnames (which means node names
    # MUST be present in /etc/hosts)
    start()

    # Get the configuration
    node_configs = Application.get_env(:artie_main, :nodes)

    # Connect to each one and keep a reference to its python process in state
    state = connect_to_all_nodes(node_configs, [])

    {:ok, state}
  end

  # TODO: This is not guaranteed to be called, and we should instead
  # really think about how to do this whole application using supervision
  def terminate(reason, state) do
    # 'state' should be a keyword list of node names to pypids so we can terminate each one
    state |> Enum.each(fn {nodename, pypid} -> shutdown(nodename, pypid) end)

    # Stop ourselves last
    stop()

    # Show the user why we are stopping
    IO.puts "Stopping Artie Infant. Reason: #{reason}"
  end


  ############## Private Functions ############## 

  # Connects to each node in the cluster and starts up the python process on it
  defp connect_to_all_nodes([], []) do
    IO.puts "Node configuration contains no node names. Terminating."
    System.halt(-1)
  end
  defp connect_to_all_nodes([{name, mod, opts}], state) do
    {:ok, pid} = connect_single_node(name, mod, opts)
    [{name, pid} | state]
  end
  defp connect_to_all_nodes([{name, mod, opts} | rest], state) do
    {:ok, pid} = connect_single_node(name, mod, opts)
    state = [{name, pid} | state]
    connect_to_all_nodes(rest, state)
  end
  
  defp connect_single_node(name, mod, opts) do
    # Connect this node to the node specified by 'name'
    connect name

    # Attempt to spawn a python process on the remote node
    {:ok, pid} = PyctoWrapper.start_link(name, mod, opts)

    # Alert the user to success
    IO.puts "Connected to #{inspect name} and fired up #{inspect mod} with opts #{inspect opts}"

    # Wait a moment to give the Python process a chance to spin up
    Process.sleep(2)

    # Return the pid of the PyctoWrapper (which serves as a proxy for the pid
    # containing the python process)
    {:ok, pid}
  end

  # Shuts down a given node
  defp shutdown(nodename, pypid) do
    result = Pyctopod.stop(pypid)
    result =
      case result do
        :ok ->
          IO.puts "Shutdown #{nodename} at pid #{pypid} was successful."
          :ok
        x ->
          IO.puts "Error #{x} when stopping #{nodename}'s Pyctopod server."
          :err
      end
    disconnect(nodename)
    result
  end


  # For some reason, these need to be in their own function wrappers

  defp start() do
    Node.start(@hostname, :shortnames)
  end

  defp stop() do
    Node.stop()
  end

  defp connect(name) do
    true = Node.connect name
  end

  defp disconnect(name) do
    Node.disconnect(name)
  end
end
