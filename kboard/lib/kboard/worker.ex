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
    File.read!(path)
    |> String.split("\n")
    |> Enum.map(fn(s)-> parse_line(s) end)
    |> Enum.filter(fn(ls)-> ls != [] end)
    |> convert_to_map()
    |> IO.inspect()
  end

  defp parse_line(line) do
    line
    |> String.split(",")
    |> Enum.map(fn(s)-> String.trim(s) end)
    |> Enum.map(fn(s)-> convert(s) end)
    |> Enum.filter(fn(s)-> s != "" end)
  end

  defp convert(str) do
    cond do
      str == "" -> ""
      String.match?(str, ~r/^(-)+ [0-9]+ (-)+$/) -> ""
      String.match?(str, ~r/^[0-9]+/) -> String.to_float(str)
      String.match?(str, ~r/^[a-zA-Z]+/) -> str
    end
  end

  defp convert_to_map([names|nums]) do
    # This returns a map: %{name: [values], another_name: [values]}
    rearranged = for {name, index} <- Enum.zip(names, 0..Enum.count(names)) do
      values = for num_list <- nums do
        Enum.at(num_list, index)
      end
    end
    Enum.zip(names, rearranged)
    |> Enum.into(%{})
  end

  defp print_images(dict) do
    # dict is a map: %{name: [values], another_name: [values]}
    for {name, values} <- dict do
      Plot.ylim([[0, 1]])
      Plot.title(name)
      Plot.plot([values])
      Plot.savefig(["./assets/static/images/" <> name <> ".png"])
      Plot.cla()
    end
  end
end
