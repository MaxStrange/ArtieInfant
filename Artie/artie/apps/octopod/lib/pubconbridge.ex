defmodule PubConBridge do
  @moduledoc """
  This module is the publisher to consumer bridge. That is,
  it is started by Pyctopod to handle messages that the running
  python instance publishes.

  It handles the messages by routing them to PubSub.
  """

  @doc """
  Starts the bridge.
  """
  def start() do
    pid = spawn fn -> loop() end
    {:ok, pid}
  end

  defp loop() do
    receive do
      {from, topic, msg} ->
        PubSub.publish(topic, {from, msg})
      other ->
        IO.puts "Got unexpected message: #{inspect other}"
    end
    loop()
  end
end
