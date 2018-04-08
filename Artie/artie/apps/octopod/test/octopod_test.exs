defmodule OctopodTest do
  use ExUnit.Case
  doctest Octopod

  @priv_path Application.app_dir(:octopod, "priv/test")
  @pyoptions [{:cd, to_charlist(@priv_path)} | Application.get_env(:octopod, :erlport_options)]

  test "Initializes Python Process" do
    {:ok, pypid} = Octopod.start(@pyoptions)
    :ok = Octopod.stop(pypid)
  end

  test "Can Call Builtin" do
    {:ok, pypid} = Octopod.start(@pyoptions)
    answer = Octopod.call(pypid, :operator, :add, [2, 3])
    assert answer == 5
    :ok = Octopod.stop(pypid)
  end

  test "Can Cast" do
    {:ok, pypid} = Octopod.start_cast(:test, @pyoptions)
    :ok = Octopod.cast(pypid, "hello")
    assert_receive({:ok, "hello FROM PYTHON!"})
    :ok = Octopod.stop(pypid)
  end

  test "Can Pass File to Python Synchronously" do
    {:ok, pypid} = Octopod.start(@pyoptions)

    fcontents = Path.join(@priv_path, "furelise.wav") |> File.read!()
    save_name = Octopod.call(pypid, :test_save_file_synchronously, :save_file, [fcontents])
    assert File.exists?(Path.join(@priv_path, save_name))

    File.rm(save_name)
    :ok = Octopod.stop(pypid)
  end

#  test "Can Pass File to Python" do
#    {:ok, pypid} = Octopod.start_cast(:test_save_file, @pyoptions)
#
#    fcontents = Path.join(@priv_path, "furelise.wav") |> File.read!()
#    :ok = Octopod.cast(pypid, fcontents)
#    assert_receive({:pyprocess, :ok}, 6_000)
#    fpath = Path.join(@priv_path, "saved_file0.wav")
#    assert File.exists?(fpath) == true
#    File.rm(fpath)
#
#    :ok = Octopod.stop(pypid)
#  end
end
