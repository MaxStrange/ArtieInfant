defmodule OctopodTest do
  use ExUnit.Case
  doctest Octopod

  test "Initializes Python Process" do
    options = Application.get_env(:octopod, :erlport_options)

    # Start the python instance using erlport
    {:ok, python} = Octopod.start_pyprocess(options)

    # Stop the python instance - we only want to make sure the test doesn't hit
    # any runtime errors
    :ok = :python.stop(python)
  end

#  test "Initializes Simple Python Script" do
#    path = 
#    {:ok, script_path} = Octopod.start_pyprocess(path)
#  end
end
