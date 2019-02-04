defmodule Robin.Test do
  def example_function(pid) do
    receive do
      {:add, [op1, op2]} -> send(pid, op1 + op2)
      {:mul, [op1, op2]} -> send(pid, op1 * op2)
      {:ask, []}         -> send(pid, self())
    end
    example_function(pid)
  end
end
