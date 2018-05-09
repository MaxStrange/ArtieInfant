defmodule Utils.FlatFiles do
  @moduledoc """
  This module exposes a way to recursively traverse file paths. Stolen from
  thegreatcodeadventure.com/elixir-tricks-building-a-recursive-function-to-list-all-files-in-a-directory/
  """

  def list_all(filepath) do
    File.ls(filepath) |> expand(filepath)
  end

  def stream_all(filepath) do
    File.ls(filepath) |> stream_expand(filepath)
  end

  defp expand({:ok, files}, path) do
    files |> Enum.flat_map(&list_all("#{path}/#{&1}"))
  end
  defp expand({:error, _}, path) do
    [path]
  end

  defp stream_expand({:ok, files}, path) do
    files |> Stream.flat_map(&stream_all("#{path}/#{&1}"))
  end
  defp stream_expand({:error, _}, path) do
    [path]
  end
end
