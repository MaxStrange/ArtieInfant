defmodule OctopodTest do
  use ExUnit.Case
  doctest Octopod

  @priv_path Application.app_dir(:octopod, "priv/test")
  @pyoptions [{:cd, to_charlist(@priv_path)} | Application.get_env(:octopod, :erlport_options)]

  test "Initializes Python Process" do
    # Start the python instance using erlport
    {:ok, python} = Octopod.start_pyprocess(@pyoptions)

    # Stop the python instance - we only want to make sure the test doesn't hit
    # any runtime errors
    :ok = Octopod.stop_pyprocess(python)
  end

  test "Runs Simple Python Script" do
    # Start a python process and have it execute some code loaded from a script

    {:ok, python} = Octopod.start_pyprocess(@pyoptions)
    {:ok, 18} = Octopod.execute_script(python, :test_simple_script)
    :ok = Octopod.stop_pyprocess(python)
  end

  test "Runs Simple Python Script with Args" do
    # Start a python process and pass it some values to add together

    {:ok, python} = Octopod.start_pyprocess(@pyoptions)
    {:ok, 312} = Octopod.execute_script(python, :test_add_nums, [300, 12])
    :ok = Octopod.stop_pyprocess(python)
  end

  test "Runs Simple Python Script with File as Arg" do
    # Start a python process and pass a serialized WAV file to it - it will save it under a different name

    {:ok, python} = Octopod.start_pyprocess(@pyoptions)
    wav_contents = Path.join(@priv_path, "furelise.wav") |> File.read!()
    {:ok, 'saved!'} = Octopod.execute_script(python, :save_file, [wav_contents])
    :ok = Octopod.stop_pyprocess(python)
  end

  test "Runs Python Script with While Loop" do
    # Start a python process that sits around in a while loop, then kill it

    {:ok, python} = Octopod.spin_script(:while, [], @pyoptions)
    Process.sleep(1000)
    Octopod.stop_pyprocess(python)
  end

  test "Can Pass Lots of Files to While Loop Script" do
    # Start a python process that sits around saving files to disk as it gets them

    wav_contents = Path.join(@priv_path, "furelise.wav") |> File.read!()
    {:ok, python} = Octopod.spin_script(:while_listen, [self()], @pyoptions)

    Process.sleep(100)
    Octopod.cast(python, wav_contents)
    Process.sleep(100)
    Octopod.cast(python, wav_contents)
    Process.sleep(100)
    Octopod.cast(python, wav_contents)

    Octopod.stop_pyprocess(python)

    assert File.exists?(Path.join(@priv_path, "savefiletest0.wav")) == true
    assert File.exists?(Path.join(@priv_path, "savefiletest1.wav")) == true
    assert File.exists?(Path.join(@priv_path, "savefiletest2.wav")) == true
  end
end
