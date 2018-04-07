defmodule OctopodTest do
  use ExUnit.Case
  doctest Octopod

  test "Initializes Python Process" do
    path = to_char_list(Path.expand("python"))
    options = [{:cd, :code.priv_dir(:octopod)},
               {:compressed, 5},
               {:start_timeout, 5},
               {:python_path, path},
               {:python, 'python'},
              ]

    # Start the python instance using erlport
    {:ok, python} = :python.start(options)

    # Stop the python instance - we only want to make sure the test doesn't hit
    # any runtime errors
    :ok = :python.stop(python)
  end

#  test "Initializes Simple Python Script" do
#    path = 
#    {:ok, script_path} = Octopod.start_pyprocess(path)
#  end

  test "greets the world" do
    assert Octopod.hello() == :world
  end
end
