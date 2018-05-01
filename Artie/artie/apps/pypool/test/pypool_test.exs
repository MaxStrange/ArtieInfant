defmodule PypoolTest do
  use ExUnit.Case
  doctest Pypool
  @masternode :"master@localhost"

  defmacro assert_next_receive(pattern, timeout \\ 100) do
    quote do
      receive do
        message ->
          assert unquote(pattern) = message
      after unquote(timeout) ->
        raise "timeout" # you might want to raise a better message here
      end
    end
  end


  test "Can Start Pypool Supervisor" do
    {:ok, pid} = Pypool.start_link(@masternode)
    Pypool.stop(pid)
  end

  test "Can Execute One Synchronous Python Command" do
    {:ok, pid} = Pypool.start_link(@masternode)
    14 = Pypool.execute_in_python(@masternode, :operator, :mul, [7, 2])
    Pypool.stop(pid)
  end

  test "Can Execute One Synchronous Python Command from Custom Module" do
    {:ok, pid} = Pypool.start_link(@masternode)
    modpath = Application.app_dir(:pypool, "priv/test") |> String.to_charlist()
    opts = [python_path: [modpath]]
    'OK' = Pypool.execute_in_python(@masternode, :testmod, :test, [], opts)
    Pypool.stop(pid)
  end

  test "Can Do Lengthy Operation in Python" do
    {:ok, pid} = Pypool.start_link(@masternode)
    modpath = Application.app_dir(:pypool, "priv/test") |> String.to_charlist()
    opts = [python_path: [modpath]]
    n = 200_000_000
    ^n = Pypool.execute_in_python(@masternode, :testmod, :count, [n], opts)
    Pypool.stop(pid)
  end

  test "Can Do Async" do
    {:ok, pid} = Pypool.start_link(@masternode)
    modpath = Application.app_dir(:pypool, "priv/test") |> String.to_charlist()
    opts = [python_path: [modpath]]
    n = 200_000_000
    mypid = self()

    spawn(fn -> send(mypid,
                Pypool.execute_in_python(@masternode, :testmod, :count, [n], opts))
          end)
    assert_receive(^n, 20_000)
    Pypool.stop(pid)
  end

  test "Can Do Async Task" do
    {:ok, pid} = Pypool.start_link(@masternode)
    modpath = Application.app_dir(:pypool, "priv/test") |> String.to_charlist()
    opts = [python_path: [modpath]]
    n = 200_000_000
    t = Task.async(fn ->
                    Pypool.execute_in_python(@masternode, :testmod, :count, [n], opts)
                   end)
    ^n = Task.await(t, 20_000)
    Pypool.stop(pid)
  end

  test "Can Only Use N Python Commands at a Time" do
    {:ok, pid} = Pypool.start_link(@masternode, 1, 0)
    modpath = Application.app_dir(:pypool, "priv/test") |> String.to_charlist()
    opts = [python_path: [modpath]]

    n = 500_000_000
    f = 10_000
    me = self()
    spawn(fn -> send(me,
                     Pypool.execute_in_python(@masternode, :testmod, :count, [n], opts))
          end)
    spawn(fn -> send(me,
                     Pypool.execute_in_python(@masternode, :testmod, :count, [f], opts))
          end)

    assert_next_receive(^n, 30_000)
    assert_next_receive(^f, 1_000)

    Pypool.stop(pid)
  end

  test "Can Use Two Different Pypools" do
    {:ok, pidA} = Pypool.start_link(@masternode)
    {:ok, pidB} = Pypool.start_link(:slavepool)

    resultA = Pypool.execute_in_python(@masternode, :operator, :add, [5, 1])
    resultB = Pypool.execute_in_python(:slavepool, :operator, :add, [4, 1])

    assert resultA == 6
    assert resultB == 5

    Pypool.stop(pidA)
    Pypool.stop(pidB)
  end
end
