defmodule OctopodTestTest do
  use ExUnit.Case
  doctest OctopodTest

  test "greets the world" do
    assert OctopodTest.hello() == :world
  end
end
