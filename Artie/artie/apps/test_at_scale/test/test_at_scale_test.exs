defmodule TestAtScaleTest do
  use ExUnit.Case
  doctest TestAtScale

  @hostname :"foo@localhost"

  setup_all do
    {:ok, _pid} = Node.start(@hostname)
    :ok
  end

  setup do
    Node.stop()
    {:ok, _pid} = Node.start(@hostname)
    :ok
  end

  test "Can Spawn A and B on Same Host" do
    mypid = self()
    _pypid0 = Node.spawn_link(@hostname, fn -> Pyctopod.start(:pyctotest_consume, nil, mypid) end)
    Process.sleep(3_000)  # Spawning is asynchronous, so we may publish before we have subscribed
    _pypid1 = Node.spawn_link(@hostname, fn -> Pyctopod.start(:pyctotest_pub_one_msg) end)

    assert_receive({:pyctotest_consume, :test, "This is a Test FROM PYTHON!"}, 15_000)

    Node.stop()
  end

  test "Can Spawn A and B on Different Hosts" do
    hostA = @hostname
    hostB = :"B@archbox"
    true = Node.connect(hostB)
    pid = Node.spawn(hostB, fn -> IO.puts "HEY" end)
    assert(is_pid(pid) == true)
    Node.stop()
  end

  test "Can Spawn Pyctopod A and B on Different Hosts" do
    hostA = @hostname
    hostB = :"B@10.1.59.187"

    mypid = self()
    Node.spawn_link(hostA, fn -> Pyctopod.start(:pyctotest_consume, nil, mypid) end)
    Process.sleep(5_000)
    Node.spawn_link(hostB, fn -> Pyctopod.start(:pyctotest_pub_one_msg) end)
    
    assert_receive({:pyctotest_consume, :test, "This is a Test FROM PYTHON!"}, 15_000)

    Node.stop()
  end
end
