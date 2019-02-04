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

  test "Can Publish a Message from A to B Manually" do
    {:ok, pypid0} = Pyctopod.start(:pyctotest_consume, self())
    {:ok, pypid1} = Pyctopod.start(:pyctotest_pub_one_msg, self())

    receive do
      {:pyctotest_pub_one_msg, :test_topic, "This is a Test"} ->
          Octopod.cast(pypid0, {:pyctotest_pub_one_msg, :test_topic, "This is a Test"})
    after
      4_000 -> assert_receive({:pyctotest_pub_one_msg, :test_topic, "This is a Test"}, 1_000)
    end
    assert_receive({:pyctotest_consume, :test, "This is a Test FROM PYTHON!"}, 5_000)

    :ok = Pyctopod.stop(pypid0)
    :ok = Pyctopod.stop(pypid1)
  end

  test "Can Publish a Message from A to B using PubSub" do
    {:ok, pypid0} = Pyctopod.start(:pyctotest_consume)
    {:ok, pypid1} = Pyctopod.start(:pyctotest_pub_one_msg)

    assert_receive({:pyctotest_consume, :test, "This is a Test FROM PYTHON!"}, 5_000)

    :ok = Pyctopod.stop(pypid0)
    :ok = Pyctopod.stop(pypid1)
  end

  @tag :remote
  test "Can Publish a Message from A to B on Remote Host using PubSub" do
    hostname = :"foo@localhost"
    {:ok, _pid} = Node.start(hostname)
    mypid = self()
    _pypid0 = Node.spawn_link(hostname, fn -> Pyctopod.start(:pyctotest_consume, nil, mypid) end)
    Process.sleep(3_000)  # Spawning is asynchronous, so we may publish before we have subscribed
    _pypid1 = Node.spawn_link(hostname, fn -> Pyctopod.start(:pyctotest_pub_one_msg) end)

    assert_receive({:pyctotest_consume, :test, "This is a Test FROM PYTHON!"}, 30_000)
    Node.stop()
  end

  test "Can Publish a Message from A to B and C using PubSub" do
    {:ok, pypid0} = Pyctopod.start(:pyctotest_consume)
    {:ok, pypid1} = Pyctopod.start(:pyctotest_consume)
    {:ok, pypid2} = Pyctopod.start(:pyctotest_pub_one_msg)

    assert_receive({:pyctotest_consume, :test, "This is a Test FROM PYTHON!"}, 5_000)
    assert_receive({:pyctotest_consume, :test, "This is a Test FROM PYTHON!"}, 5_000)

    :ok = Pyctopod.stop(pypid0)
    :ok = Pyctopod.stop(pypid1)
    :ok = Pyctopod.stop(pypid2)
  end

  test "Can Pass File from A to B using PubSub" do
    {:ok, pypid0} = Pyctopod.start(:pyctotest_consume_file)
    {:ok, pypid1} = Pyctopod.start(:pyctotest_pub_file)

    assert_receive({:pyctotest_consume_file, :test, fname}, 15_000)
    fpath = [Application.app_dir(:octopod, "priv/test"), fname] |> Path.join()
    assert(File.exists?(fpath) == true)
    File.rm!(fpath)

    :ok = Pyctopod.stop(pypid0)
    :ok = Pyctopod.stop(pypid1)
  end

  test "Can Pass File from A to B and Have it Come out the Same" do
    {:ok, pypid0} = Pyctopod.start(:pyctotest_consume_file)
    {:ok, pypid1} = Pyctopod.start(:pyctotest_pub_file)

    assert_receive({:pyctotest_consume_file, :test, fname}, 15_000)

    fpath = [Application.app_dir(:octopod, "priv/test"), fname] |> Path.join()
    original = [Application.app_dir(:octopod, "priv/test"), "furelise.wav"] |> Path.join()

    assert(File.exists?(fpath) == true)

    # Checksum
    hash_original = 
      File.stream!(original, [], 2048) 
      |> Enum.reduce(:crypto.hash_init(:sha256), fn(line, acc) -> :crypto.hash_update(acc,line) end ) 
      |> :crypto.hash_final 
      |> Base.encode16 
    hash_new =
      File.stream!(fpath, [], 2048) 
      |> Enum.reduce(:crypto.hash_init(:sha256), fn(line, acc) -> :crypto.hash_update(acc,line) end ) 
      |> :crypto.hash_final 
      |> Base.encode16 

    assert(hash_original == hash_new)

    File.rm!(fpath)

    :ok = Pyctopod.stop(pypid0)
    :ok = Pyctopod.stop(pypid1)
  end

  @tag :lengthy
  @tag timeout: 200_000
  test "Pyctopod Does Not Exit While Elixir is Still Around" do
    {:ok, pypid0} = Pyctopod.start(:pyctotest_keepalive_receiver)
    {:ok, pypid1} = Pyctopod.start(:pyctotest_keepalive_sender)

    assert_receive({:pyctotest_keepalive_receiver, :test, _msg}, 60_000)
    assert_receive({:pyctotest_keepalive_receiver, :test, _msg}, 60_000)
    assert_receive({:pyctotest_keepalive_receiver, :test, _msg}, 60_000)

    :ok = Pyctopod.stop(pypid0)
    :ok = Pyctopod.stop(pypid1)
  end

  test "Can Pass File from A to B and C" do
    {:ok, pypid0} = Pyctopod.start(:pyctotest_consume_file)
    {:ok, pypid2} = Pyctopod.start(:pyctotest_consume_file)
    {:ok, pypid1} = Pyctopod.start(:pyctotest_pub_file)

    assert_receive({:pyctotest_consume_file, :test, fname0}, 15_000)
    assert_receive({:pyctotest_consume_file, :test, fname1}, 25_000)

    fpath0 = [Application.app_dir(:octopod, "priv/test"), fname0] |> Path.join()
    fpath1 = [Application.app_dir(:octopod, "priv/test"), fname1] |> Path.join()
    original = [Application.app_dir(:octopod, "priv/test"), "furelise.wav"] |> Path.join()

    assert(File.exists?(fpath0) == true)
    assert(File.exists?(fpath1) == true)

    # Checksum
    hash_original = 
      File.stream!(original, [], 2048) 
      |> Enum.reduce(:crypto.hash_init(:sha256), fn(line, acc) -> :crypto.hash_update(acc,line) end ) 
      |> :crypto.hash_final 
      |> Base.encode16 
    hash_new0 =
      File.stream!(fpath0, [], 2048) 
      |> Enum.reduce(:crypto.hash_init(:sha256), fn(line, acc) -> :crypto.hash_update(acc,line) end ) 
      |> :crypto.hash_final 
      |> Base.encode16 
    hash_new1 =
      File.stream!(fpath1, [], 2048) 
      |> Enum.reduce(:crypto.hash_init(:sha256), fn(line, acc) -> :crypto.hash_update(acc,line) end ) 
      |> :crypto.hash_final 
      |> Base.encode16 

    assert(hash_original == hash_new0)
    assert(hash_original == hash_new1)

    File.rm!(fpath0)
    File.rm!(fpath1)

    :ok = Pyctopod.stop(pypid0)
    :ok = Pyctopod.stop(pypid1)
    :ok = Pyctopod.stop(pypid2)
  end

  test "Can Add to Python Path" do
    path = Application.app_dir(:octopod, "priv")
    {:ok, pypid} = Pyctopod.start(:pyctotest_simple_msg_dif_loc, self(), nil, [python_path: [path]])

    assert_receive({:tester, :test_topic, "This is a Test"}, 5_000)

    :ok = Pyctopod.stop(pypid)
  end
end
