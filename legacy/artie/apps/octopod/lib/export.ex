defmodule Octopod.Export do
  @moduledoc """
  Provides a more comfortable Elixir interface to the erlport library. This should be
  the only module that references erlport stuff directly.
  """

  @doc """
  Start the python instance
  """
  def start(pyargs) do
    {:ok, pid} = :python.start(pyargs)
    pid
  end

  def call(pid, mod, func, args \\ []) do
    :python.call(pid, mod, func, args)
  end

  def cast(pid, msg) do
    :python.cast(pid, msg)
  end

  def stop(pid) do
    :python.stop(pid)
  end
end
