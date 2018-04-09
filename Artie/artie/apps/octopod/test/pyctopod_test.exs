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

  test "Can Spawn Two Processes" do
    {:ok, pypid0} = Pyctopod.start(:pyctotest, self())
    {:ok, pypid1} = Pyctopod.start(:pyctotest, self())

    Process.sleep(1)

    :ok = Pyctopod.stop(pypid0)
    :ok = Pyctopod.stop(pypid1)
  end

  test "Can Publish a Message from A to B" do
    {:ok, pypid0} = Pyctopod.start(:pyctotest_consume, self())
    {:ok, pypid1} = Pyctopod.start(:pyctotest_pub_one_msg, self())

    assert_receive({:pyctotest_consume, :test, "This is a Test FROM PYTHON!"}, 5_000)

    :ok = Pyctopod.stop(pypid0)
    :ok = Pyctopod.stop(pypid1)
  end
end
