defmodule PyctopodTest do
  use ExUnit.Case

  test "Can Start a module that Uses Pyctopod" do
    {:ok, pypid} = Pyctopod.start()
    :ok = Pyctopod.stop(pypid)
  end
end
