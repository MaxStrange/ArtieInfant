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

    assert_receive({:pyctotest_consume_file, :test, fname}, 5_000)
    fpath = [Application.app_dir(:octopod, "priv/test"), fname] |> Path.join()
    assert(File.exists?(fpath) == true)
    File.rm!(fpath)

    :ok = Pyctopod.stop(pypid0)
    :ok = Pyctopod.stop(pypid1)
  end

  test "Can Pass File from A to B and Have it Come out the Same" do
    {:ok, pypid0} = Pyctopod.start(:pyctotest_consume_file)
    {:ok, pypid1} = Pyctopod.start(:pyctotest_pub_file)

    assert_receive({:pyctotest_consume_file, :test, fname}, 5_000)

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
      File.stream!(original, [], 2048) 
      |> Enum.reduce(:crypto.hash_init(:sha256), fn(line, acc) -> :crypto.hash_update(acc,line) end ) 
      |> :crypto.hash_final 
      |> Base.encode16 

    assert(hash_original == hash_new)

    File.rm!(fpath)

    :ok = Pyctopod.stop(pypid0)
    :ok = Pyctopod.stop(pypid1)
  end
end
