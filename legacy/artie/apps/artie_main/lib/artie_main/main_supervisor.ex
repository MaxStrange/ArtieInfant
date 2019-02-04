defmodule ArtieMain.MainSupervisor do
  @moduledoc """
  This module is a supervisor for the main application
  module that runs on the master node (the computer you are
  invoking this application from).
  """
  use Supervisor

  def start_link(args) do
    Logger.enable(self())
    Supervisor.start_link(__MODULE__, args, name: __MODULE__)
  end

  def init(_args) do
    robin_server = Application.get_env(:artie_main, :function_node_map) |> ArtieMain.Robin.Server.start_link()
    children = [
      %{
        id: ArtieMain,
        start: {ArtieMain, :start_link, [robin_server]}
      },
    ]
    opts = [strategy: :one_for_one, name: __MODULE__]
    Supervisor.init(children, opts)
  end
end
