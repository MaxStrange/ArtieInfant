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
  def stop_pyprocess(pypid) do
    :python.stop(pypid)
  end
  def stop_pyprocess(pypid, pid) do
    :python.stop(pypid)
    Process.exit(pid, :kill)
    :ok
  end

  @doc """
  Executes the given python script and returns :ok or {:err, stack_trace}.

  *NOTE* The script must be local to the python process (use the {:cd, directory} option
  with start_pyprocess) or must be in the path that python uses to load modules. Also,
  the script must have a main() function that takes no arguments.

  Returns whatever main() returns, synchronously.

  For asynchronously starting a python process, see Octopod.spin_script/3.

  The below examples assume a test_doctest.py script that contains the following code:

  def main():
    return 5 + 5

  ## Examples

    iex> path = 'C:/Users/maxst/repos/ArtieInfant/Artie/artie/apps/octopod/priv/test'
    iex> {:ok, pid} = Octopod.start_pyprocess([{:cd, path}])
    iex> Octopod.execute_script(pid, :test_doctest)
    {:ok, 10}

  """
  def execute_script(pyproc, module, args \\ []) do
    result = :python.call(pyproc, module, :main, args)
    {:ok, result}
  end

  @doc """
  Executes the given python module by spawning a new process to run it in. Returns the
  pid of that process, which can be used in all the functions in this module, just like
  the pid returned by any of the start_* functions.

  This results in an asynchronous execution of the python module's main() function.

  Parameters:
    module:     The module, which must be in the python path.
    args:       The arguments to the main() function.
    pyoptions:  The options to pass to :python.start_link/1

  Returns:
    {:ok, pid}, where `pid` can be used just like the pids returned by any of the start_*
    functions in this module.

  ## Examples

    iex> path = 'C:/Users/maxst/repos/ArtieInfant/Artie/artie/apps/octopod/priv/test'
    iex> {:ok, pypid} = Octopod.spin_script(:while, [], [{:cd, path}])
    iex> Octopod.stop_pyprocess(pypid)
    :ok

  """
  def spin_script(module, args \\ [], pyoptions \\ []) do
    {:ok, pyproc} = :python.start(pyoptions)
    spawn fn -> spin(pyproc, module, args) end
    {:ok, pyproc}
  end

  defp spin(pyproc, module, args) do
    ospid = :python.call(pyproc, :os, :getpid, [])
    {kill_cmd, kill_args} = get_kill_cmd(ospid)
    try do
      :python.call(pyproc, module, :main, args, [{:timeout, :infinity}])
    rescue
      ErlangError -> System.cmd(kill_cmd, kill_args)
    end
  end

  defp get_kill_cmd(pid) do
    case :os.type() do
      {:win32, _} -> {"TaskKill", ["/PID", to_string(pid), "/F"]}
      {:unix, _} -> {"kill", ["-9", pid]}
      _ -> IO.puts "You may need to manually kill the process with PID" <> to_string(pid)
    end
  end

  @doc """
  Writes `msg` to `pyproc` via erlport's cast/2 function. This means that the running
  python process must have been created via an asynchronous mechanism such as this module's
  spin_script *and* the python module must have registered a handler with erlport as:

  ```python
  from erlport.erlterms import Atom
  import erlport.erlang as erl

  def register_handler(dest):
    def handler(msg):
      erl.cast(dest, msg)
    erl.set_message_handler(handler)
    return Atom("ok")

  # You should give self() or some other pid as dest
  ```

  ## Examples

    iex> path = 'C:/Users/maxst/repos/ArtieInfant/Artie/artie/apps/octopod/priv/test'
    iex> {:ok, pypid} = Octopod.spin_script(:while_echo, [self()], [{:cd, path}])
    iex> :ok = Octopod.cast(pypid, 'Here is a message')
    iex> :ok = Process.sleep(100)
    iex> receive do
    ...>   "Here is a message FROM PYTHON!" -> :ok
    ...>   _ -> :err
    ...> after
    ...>   1_000 -> :err_timeout
    ...> end
    :ok
    iex> Octopod.stop_pyprocess(pypid)
    :ok

  """
  def cast(pyproc, msg) do
    :python.cast(pyproc, msg)
  end
end
