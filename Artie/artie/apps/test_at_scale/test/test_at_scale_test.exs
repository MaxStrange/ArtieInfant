defmodule TestAtScaleTest do
  use ExUnit.Case
  doctest TestAtScale

  test "greets the world" do
    assert TestAtScale.hello() == :world
  end

  test "Can Spawn Processes on Same Host" do
    hostname = :"foo@localhost"
    {:ok, _pid} = Node.start(hostname)
    mypid = self()
    _pypid0 = Node.spawn_link(hostname, fn -> Pyctopod.start(:pyctotest_consume, nil, mypid) end)
    Process.sleep(3_000)  # Spawning is asynchronous, so we may publish before we have subscribed
    _pypid1 = Node.spawn_link(hostname, fn -> Pyctopod.start(:pyctotest_pub_one_msg) end)

    assert_receive({:pyctotest_consume, :test, "This is a Test FROM PYTHON!"}, 30_000)
  end
end
