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

  test "Can Pass File to Python Asynchronously" do
    {:ok, pypid} = Octopod.start_cast(:test_save_file, @pyoptions)

    fcontents = Path.join(@priv_path, "furelise.wav") |> File.read!()
    :ok = Octopod.cast(pypid, fcontents)
    assert_receive({:pyprocess, 0}, 6_000)

    fpath = Path.join(@priv_path, "saved_file0.wav")
    assert File.exists?(fpath) == true
    File.rm(fpath)
    :ok = Octopod.stop(pypid)
  end

  test "Can Pass Two Files to Python Asynchronously" do
    {:ok, pypid} = Octopod.start_cast(:test_save_file, @pyoptions)

    fcontents = Path.join(@priv_path, "furelise.wav") |> File.read!()
    :ok = Octopod.cast(pypid, fcontents)
    :ok = Octopod.cast(pypid, fcontents)
    assert_receive({:pyprocess, 0}, 6_000)
    assert_receive({:pyprocess, 1}, 6_000)

    fpath = Path.join(@priv_path, "saved_file0.wav")
    assert File.exists?(fpath) == true
    File.rm(fpath)

    fpath = Path.join(@priv_path, "saved_file1.wav")
    assert File.exists?(fpath) == true
    File.rm(fpath)

    :ok = Octopod.stop(pypid)
  end

  test "Can Pass Several Files to Python Asynchronously" do
    opts = @pyoptions |> Enum.reject(&(Enum.fetch!(Tuple.to_list(&1), 0) == :call_timeout)) |> Enum.concat([{:call_timeout, 60_000}])
    {:ok, pypid} = Octopod.start_cast(:test_save_file, opts)

    fcontents = Path.join(@priv_path, "furelise.wav") |> File.read!()
    :ok = Octopod.cast(pypid, fcontents)
    :ok = Octopod.cast(pypid, fcontents)
    :ok = Octopod.cast(pypid, fcontents)
    :ok = Octopod.cast(pypid, fcontents)
    :ok = Octopod.cast(pypid, fcontents)
    :ok = Octopod.cast(pypid, fcontents)
    :ok = Octopod.cast(pypid, fcontents)
    :ok = Octopod.cast(pypid, fcontents)
    :ok = Octopod.cast(pypid, fcontents)
    :ok = Octopod.cast(pypid, fcontents)
    assert_receive({:pyprocess, 0}, 60_000)
    assert_receive({:pyprocess, 1}, 6_000)
    assert_receive({:pyprocess, 2}, 6_000)
    assert_receive({:pyprocess, 3}, 6_000)
    assert_receive({:pyprocess, 4}, 6_000)
    assert_receive({:pyprocess, 5}, 6_000)
    assert_receive({:pyprocess, 6}, 6_000)
    assert_receive({:pyprocess, 7}, 6_000)
    assert_receive({:pyprocess, 8}, 6_000)
    assert_receive({:pyprocess, 9}, 6_000)

    fpath = Path.join(@priv_path, "saved_file0.wav")
    assert File.exists?(fpath) == true
    File.rm(fpath)

    fpath = Path.join(@priv_path, "saved_file1.wav")
    assert File.exists?(fpath) == true
    File.rm(fpath)

    fpath = Path.join(@priv_path, "saved_file2.wav")
    assert File.exists?(fpath) == true
    File.rm(fpath)

    fpath = Path.join(@priv_path, "saved_file3.wav")
    assert File.exists?(fpath) == true
    File.rm(fpath)

    fpath = Path.join(@priv_path, "saved_file4.wav")
    assert File.exists?(fpath) == true
    File.rm(fpath)

    fpath = Path.join(@priv_path, "saved_file5.wav")
    assert File.exists?(fpath) == true
    File.rm(fpath)

    fpath = Path.join(@priv_path, "saved_file6.wav")
    assert File.exists?(fpath) == true
    File.rm(fpath)

    fpath = Path.join(@priv_path, "saved_file7.wav")
    assert File.exists?(fpath) == true
    File.rm(fpath)

    fpath = Path.join(@priv_path, "saved_file8.wav")
    assert File.exists?(fpath) == true
    File.rm(fpath)

    fpath = Path.join(@priv_path, "saved_file9.wav")
    assert File.exists?(fpath) == true
    File.rm(fpath)

    :ok = Octopod.stop(pypid)

  end
end
