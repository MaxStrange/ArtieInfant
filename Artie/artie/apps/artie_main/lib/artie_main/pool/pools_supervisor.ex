defmodule ArtieMain.PoolsSupervisor do
  @moduledoc """
  Supervisor responsible for starting a Pypool node on each physical
  node in the system.
  """
  use Supervisor

  def start_link(nodes) do
    Logger.enable(self())
    Supervisor.start_link(__MODULE__, nodes, name: __MODULE__)
  end

  def init(nodes) do
    Printex.prints "Attempting to connect to nodes...", :light_blue
    nodes
    |> Enum.map(fn n -> {Node.connect(n), n} end)
    |> Enum.each(fn {result, n} -> print_results(n, result) end)

    children =
      nodes
      |> Enum.map(fn n -> {Atom.to_string(n), Atom.to_string(n)} end)
      |> Enum.map(fn {n, m} -> {n, n <> "_pypool", m <> "_pool_supervisor"} end)
      |> Enum.map(fn {node, n, m} -> {String.to_atom(node), String.to_atom(n), String.to_atom(m)} end)
      |> Enum.map(fn {node, n, m} -> %{id: n, start: {ArtieMain.PoolnodeStarter, :start_link, [node, n, m]}, shutdown: 100, type: :supervisor} end)

    opts = [strategy: :one_for_one, name: ArtieMain.PoolsSupervisor, max_restarts: 1]
    Supervisor.init(children, opts)
  end

  defp print_results(node, result) do
    if result do
      Printex.prints "  -> #{inspect node}: #{inspect result}", :light_green
    else
      Printex.prints "  -> #{inspect node}: #{inspect result}", :light_red
    end
  end
end
