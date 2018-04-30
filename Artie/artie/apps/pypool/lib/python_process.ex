defmodule Pypool.PythonProcess do
  @moduledoc """
  This module is a GenServer representation of a Python Process.

  Calling GenServer.call() on this module will return the result of the
  given python operation.
  """
  use GenServer

  def start_link(_) do
    GenServer.start_link(__MODULE__, nil, [])
  end

  @doc """
  Parameters:
    - module: The python module.
    - func:   The python function found in the module.
    - args:   A list of arguments to pass to the function.
    - opts:   Erlport opts (a keyword list).

  ## Examples

    iex> {:ok, pid} = Pypool.PythonProcess.start_link(nil)
    iex> Pypool.PythonProcess.call(pid, {:operator, :add, [4, 2], []})
    6

  """
  def call(pid, {module, func, args, opts}, timeout \\ 5_000) do
    GenServer.call(pid, {module, func, args, opts}, timeout)
  end


  ############ Callbacks ############ 

  def init(_) do
    {:ok, nil}
  end

  def handle_call({module, func, args, []}, _from, state) do
    pid = Pypool.Export.start()
    handle_call_base(pid, module, func, args, state)
  end
  def handle_call({module, func, args, opts}, _from, state) do
    pid = Pypool.Export.start(opts)
    handle_call_base(pid, module, func, args, state)
  end
  defp handle_call_base(pid, module, func, args, state) do
    result = Pypool.Export.call(pid, module, func, args)
    Pypool.Export.stop(pid)
    {:reply, result, state}
  end
end
