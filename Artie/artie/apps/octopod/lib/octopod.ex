defmodule Octopod do
  @moduledoc """
  This module is the API for the library.

  The way this library works is this:
  1. Your elixir application starts a python instance that runs whatever python code
  2. That python code will want to send files to other runnin python instances
  3. The python code calls into this library (perhaps through a wrapper lib), 'send(fpath, topic)'
  4. This library finds the file specified and sends it (perhaps with compression) to anyone listening to 'topic'
  5. The listening servers receive the file, deserialize it, and hand it over to the python process that they are running

  """

  @doc """
  Starts a python process. Just a wrapper for :python.start/0.

  ## Examples

      iex> {:ok, pid} = Octopod.start_pyprocess()
      iex> is_pid(pid)
      true

  """
  def start_pyprocess do
    :python.start()
  end

  @doc """
  Starts a python process. Just a wrapper for :python.start/1.

  ## Examples

    iex> {:ok, pid} = Octopod.start_pyprocess([{:compressed, 5}])
    iex> is_pid(pid)
    true

  """
  def start_pyprocess(options) do
    :python.start(options)
  end

  @doc """
  Stops the given python process.

  ## Examples

    iex> {:ok, pid} = Octopod.start_pyprocess()
    iex> Octopod.stop_pyprocess(pid)
    :ok

  """
  def stop_pyprocess(pid) do
    :python.stop(pid)
  end

  @doc """
  Executes the given python script and returns :ok or {:err, stack_trace}.

  *NOTE* The script must be local to the python process (use the {:cd, directory} option
  with start_pyprocess) or must be in the path that python uses to load modules. Also,
  the script must have a main() function that takes no arguments.

  Returns whatever main() returns.

  The below examples assume a test_doctest.py script that contains the following code:

  def main():
    return 5 + 5

  ## Examples

    iex> path = 'C:/Users/maxst/repos/ArtieInfant/Artie/artie/apps/octopod/priv'
    iex> {:ok, pid} = Octopod.start_pyprocess([{:cd, path}])
    iex> Octopod.execute_script(pid, :test_doctest)
    {:ok, 10}

  """
  def execute_script(pyproc, module, args \\ []) do
    result = :python.call(pyproc, module, :main, args)
    {:ok, result}
  end
end
