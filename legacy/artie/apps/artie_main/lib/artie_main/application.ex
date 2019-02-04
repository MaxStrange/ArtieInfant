defmodule ArtieMain.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  def start(_type, _args) do
    Logger.enable(self())
    Supervisor.start_link(__MODULE__, :ok, name: ArtieMain.Application)
  end

  def init(:ok) do
    # List all child processes to be supervised
    nodes = Application.get_env(:artie_main, :nodenames)
    children = [
      # Starts a worker by calling: ArtieMain.Worker.start_link(arg)
      # {ArtieMain.Worker, arg},
      {ArtieMain.PoolsSupervisor, nodes},
      {ArtieMain.MainSupervisor, nil}
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: ArtieMain.Supervisor, max_restarts: 1]
    Supervisor.init(children, opts)
  end
end
