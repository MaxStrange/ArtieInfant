defmodule ArtieMain.Robin.Helper do
  @moduledoc """
  This module defines the function that will be run in each Robin.
  """

  @doc """
  This is the function that runs on each Robin instance.
  """
  def loop_func(pid) do
    receive do
      {:apply_func, pymod, pyfunc, pyargs} ->
        node = Node.self()
        result = Pypool.execute_in_python(node, pymod, pyfunc, pyargs)
        send(pid, result)
    end
    loop_func(pid)
  end
end
