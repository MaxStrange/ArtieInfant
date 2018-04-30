defmodule ArtieMainTest do
  use ExUnit.Case
  doctest ArtieMain

  test "greets the world" do
    assert ArtieMain.hello() == :world
  end
end
