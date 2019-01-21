defmodule ArtieMain do
  @moduledoc """
  This is the main module for Artie.
  """
  use GenServer

  def start_link(robin_server) do
    GenServer.start_link(__MODULE__, robin_server, name: __MODULE__)
  end

  def init(robin_server) do
    {:ok, robin_server}
  end

  def handle_cast(:start, robin_server) do
    spawn_link(fn -> main(robin_server) end)
    {:noreply, robin_server}
  end

  defp pyrpc(wav, robin_server, func, args) do
    robin = GenServer.call(robin_server, {:lookup, func})
    {m, f} = Application.get_env(:artie_main, :func_to_pymf) |> Map.fetch!(func)
    msg = {:execute, robin, m, f, [wav] ++ args}
    GenServer.call(robin_server, msg)
  end

  defp main(robin_server) do
    Application.get_env(:artie_main, :datapath)
    |> Utils.FlatFiles.stream_all()
    |> Stream.each(fn fp -> spawn(fn -> test(robin_server, fp) end) end)
    |> Stream.run()
    #main() <-- to keep alive
  end

  defp test(robin_server, filepath) do
    is_wav = filepath |> Path.extname() |> String.downcase() |> String.equivalent?(".wav")
    if is_wav do
      filepath
      |> pyrpc(robin_server, :load_from_file, [])
      |> pyrpc(robin_server, :remove_silence, [])
    end
  end
end
