defmodule OctopodTest do
  use ExUnit.Case
  doctest Octopod

  @pyoptions Application.get_env(:octopod, :erlport_options)

  test "Initializes Python Process" do
    # Start the python instance using erlport
    {:ok, python} = Octopod.start_pyprocess(@pyoptions)

    # Stop the python instance - we only want to make sure the test doesn't hit
    # any runtime errors
    :ok = Octopod.stop_pyprocess(python)
  end

  test "Runs Simple Python Script" do
    # Start a python process and have it execute some code loaded from a script

    priv_path = Application.app_dir(:octopod, "priv")
    {:ok, python} = Octopod.start_pyprocess([{:cd, to_charlist(priv_path)} | @pyoptions])

    {:ok, 18} = Octopod.execute_script(python, :test_simple_script)

    :ok = Octopod.stop_pyprocess(python)
  end
end