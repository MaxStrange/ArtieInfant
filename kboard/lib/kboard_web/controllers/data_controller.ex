defmodule KboardWeb.DataController do
  use KboardWeb, :controller

  def data(conn, params) do
    metric = params["metric"]
    {x, _} = params["x"] |> Integer.parse()
    d = parse_log_file(x, metric)
    json conn, d
  end

  @path "/home/max/git_repos/ArtieInfant/scratch/mlpvad/log.csv"

  defp parse_log_file(x, metric) do
    File.read!(@path)
    |> String.split("\n")
    |> Enum.map(fn(s)-> parse_line(s) end)
    |> Enum.filter(fn(ls)-> ls != [] end)
    |> convert_to_map()
    |> get_latest_y(x, metric)
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
    rearranged = for {_name, index} <- Enum.zip(names, 0..Enum.count(names)) do
      _values = for num_list <- nums do
        Enum.at(num_list, index)
      end
    end
    Enum.zip(names, rearranged)
    |> Enum.into(%{})
  end

  defp get_latest_y(map, x, metric) do
    values = map[metric]
    cond do
      is_list(values) and Enum.count(values) > x  ->
        # Take at most 5 items at a time
        count = Enum.count(values)
        end_index = if count < 10, do: count, else: 10
        Enum.slice(values, x..x + end_index)
      is_list(values) and Enum.count(values) <= x -> nil
      true                                        -> nil
    end
  end
end
