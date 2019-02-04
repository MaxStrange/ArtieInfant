defmodule TestAtScale do
  @moduledoc """
  """
  @hostname :"master@localhost"

  def main(_args) do
    start()
    nodenames = Application.get_env(:test_at_scale, :nodelist)
    connect_all_nodes(nodenames)

    ##TODO###
    {:ok, pypid0} = Pyctopod.start(:pyctotest, self())
    {:ok, pypid1} = Pyctopod.start(:pyctotest, self())

    Process.sleep(1)

    :ok = Pyctopod.stop(pypid0)
    :ok = Pyctopod.stop(pypid1)
    #########
  end

  defp connect_all_nodes([name]) do
    connect name
  end
  defp connect_all_nodes([name | node_names]) do
    connect name
    connect_all_nodes(node_names)
  end
  defp connect_all_nodes([]) do
    IO.puts "Need a list of nodes to connect to."
    stop()
    System.halt(0)
  end

  # For reasons unknown to me, these functions must be in their own
  # wrapper functions to work. At the very least, this is definitely
  # true for connect()
  defp start() do
    Node.start(@hostname, :shortnames)
  end

  defp stop() do
    Node.stop()
  end

  defp connect(name) do
    true = Node.connect name
  end
end
