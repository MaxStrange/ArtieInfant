defmodule Kboard.Worker do
  use GenServer
  import :timer
  alias Expyplot.Plot

  def start_link(name, [path]) do
    GenServer.start_link(__MODULE__, path, [name: name])
  end

  def init(path) do
    spawn_link(fn -> refresh(path) end)
    {:ok, path}
  end

  defp refresh(path) do
    parse_log_file(path) |> print_images()
    :timer.sleep(5000)
    refresh path
  end

  defp parse_log_file(path) do
    File.read(path)
  end

  defp print_images(_blah) do
    Plot.plot([[1, 2, 3]])
    Plot.show()
  end
end
