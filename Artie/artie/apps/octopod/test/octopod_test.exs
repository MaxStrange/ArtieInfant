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
    # TODO: This one next
  end

  test "Runs Simple Python Script" do
  end

  test "Runs Simple Python Script with Args" do
  end

  test "Runs Simple Python Script with File as Arg" do
  end

  test "Runs Python Script with While Loop" do
  end

  test "Can Pass Lots of Files to While Loop Script" do
  end
end
