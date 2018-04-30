defmodule Pypool do
  @moduledoc """
  The Pypool module is a supervisor that supervises exactly one instance of
  poolboy. It takes an optional poolboy configuration.

  The point of this module is to supervise a poolboy of Erlport workers - that is,
  workers that spin up a Python instance and execute some function in Python.
  A Python instance is very heavy weight, so we need to guard against making too
  many of them - hence, poolboy.
  """
  use Supervisor

  @doc """
  This is the entry point to the Pypool library. It takes an optional configuration
  for Poolboy, especially important is the maximum number of workers.
  """
  def start_link(nodename, size \\ 5, max_overflow \\ 2, strategy \\ :lifo) do
    Supervisor.start_link(__MODULE__, {nodename, size, max_overflow, strategy})
  end

  @doc """
  Executes a single synchronous request in a Python context and returns the result.

  Parameters:
    - nodename:     The name of the Pool - which should be the same as the name that you
                    passed in to start_link.
    - mod:          An atom of the python module.
    - func:         An atom of the python moduel's function.
    - args:         Optional list of args to pass to the function.
    - opts:         Optional keyword list of erlport options.
    - timeout:      Optional timeout value - defaults to infinity

  Returns:
    The result of the Python execution.

  ## Examples

    iex> {:ok, _pid} = Pypool.start_link(:"master@localhost")
    iex> Pypool.execute_in_python(:"master@localhost", :operator, :add, [9, 1])
    10

  """
  def execute_in_python(nodename, mod, func, args \\ [], opts \\ [], timeout \\ :infinity) do
    poolboy_timeout = timeout
    :poolboy.transaction(
         nodename,
         fn pid -> Pypool.PythonProcess.call(pid, {mod, func, args, opts}, timeout) end,
         poolboy_timeout)
  end

  @doc """
  Stops the given Pypool and cleans up Poolboy and any remaining Python processes.
  """
  def stop(pid) do
    Supervisor.stop(pid)
  end


  ############ Callbacks ############ 

  def init ({nodename, size, max_overflow, strategy}) do
    poolboy_config = [
      {:name, {:local, nodename}},
      {:worker_module, Pypool.PythonProcess},
      {:size, size},
      {:max_overflow, max_overflow},
      {:strategy, strategy}
    ]

    children = [
      :poolboy.child_spec(nodename, poolboy_config)
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
