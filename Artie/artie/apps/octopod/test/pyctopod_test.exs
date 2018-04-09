defmodule PyctopodTest do
  use ExUnit.Case
  doctest Pyctopod

  test "Can Start a Module that Uses Pyctopod" do
    {:ok, pypid} = Pyctopod.start(:pyctotest, self())
    :ok = Pyctopod.stop(pypid)
  end

  test "Can Publish a Message from a Pyctopod Client" do
    {:ok, pypid} = Pyctopod.start(:pyctotest_simple_msg, self())

    assert_receive({:tester, :test_topic, "This is a Test"}, 5_000)

    :ok = Pyctopod.stop(pypid)
  end
end
