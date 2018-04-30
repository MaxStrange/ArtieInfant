defmodule RobinTest do
  use ExUnit.Case
  doctest Robin
  @moduletag :remote

  defp example_function(pid) do
    receive do
      {:add, [op1, op2]} -> send(pid, op1 + op2)
      {:mul, [op1, op2]} -> send(pid, op1 * op2)
    end
    example_function(pid)
  end

  test "Can Start Robin" do
    Node.start(:"fooftar@localhost", :shortnames)
    {:ok, pid} = Robin.start_link([:"fooftar@localhost"], &example_function/1)
    assert is_pid(pid) == true
  end

  test "Can Start Two Robins" do
    Node.start(:"fooftar@localhost", :shortnames)
    {:ok, _pid1} = Robin.start_link([:"fooftar@localhost"], &example_function/1)
    {:ok, _pid2} = Robin.start_link([:"fooftar@localhost"], &example_function/1)
  end

  test "Can Use Robin to Execute Function on Node" do
    Node.start(:"fooftar@localhost", :shortnames)
    {:ok, pid} = Robin.start_link([:"fooftar@localhost"], &example_function/1)
    result = Robin.call(pid, {:add, [2, 2]})
    assert result == 4
  end

  test "Can Start Robin with a Single Remote Node" do
    Node.start(:"fooftar@localhost", :shortnames)
    remote_node = :"pi100@pi100"
    {:ok, pid} = Robin.start_link([remote_node], &Robin.Test.example_function/1)
    assert is_pid(pid) == true
  end

  test "Can Execute Function on Remote Node" do
    Node.start(:"fooftar@localhost", :shortnames)
    remote_node = :"pi100@pi100"
    {:ok, pid} = Robin.start_link([remote_node], &Robin.Test.example_function/1)
    result = Robin.call(pid, {:mul, [3, 3]})
    assert result == 9
  end

  test "Round Robin Actually Happens" do
    local = :"fooftar@localhost"
    Node.start(local, :shortnames)
    remote_node = :"pi100@pi100"
    {:ok, pid} = Robin.start_link([remote_node, local], &Robin.Test.example_function/1)

    # Ask the nodes for their pids and make sure that we cycled through them
    rresult = Robin.call(pid, {:ask, []})
    lresult = Robin.call(pid, {:ask, []})
    xresult = Robin.call(pid, {:ask, []})

    assert rresult != lresult
    assert rresult == xresult
  end
end
