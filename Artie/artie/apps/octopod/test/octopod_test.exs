defmodule OctopodTest do
  use ExUnit.Case
  doctest Octopod

  test "greets the world" do
    assert Octopod.hello() == :world
  end
end
