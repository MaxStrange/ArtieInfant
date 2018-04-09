defmodule PubConBridge do
  @moduledoc """
  This module is the publisher to consumer bridge. That is,
  it is started by Pyctopod to handle messages that the running
  python instance publishes.

  It handles the messages by routing them to PubSub.
  """

  @doc """
  Starts the bridge. `testpid` is a pid to eavesdrop on all
  messages published to :test.
  """
  def start(pyctopid, testpid \\ nil) do
    testpid = if (testpid == nil), do: self(), else: testpid
    pid = spawn fn -> loop(pyctopid, [], testpid) end
    {:ok, pid}
  end

  defp loop(pyctopid, pids, testpid) do
    {pyctopid, pids} =
      receive do
        {from, :test, msg} ->
          # For testing purposes - we listen to the test topic on our test process
          send(testpid, {from, :test, msg})
          # Since :test is a perfectly valid topic name, we will also pass it on
          PubSub.publish(:test, {from, msg})
          {pyctopid, pids}
        {from, topic, msg} ->
          PubSub.publish(topic, {from, msg})
          {pyctopid, pids}
        {:subscribe, topic} ->
          pid = spawn_link fn -> subscribe(pyctopid, topic) end
          PubSub.subscribe(pid, topic)
          {pyctopid, [pid | pids]}
        {:pyctopid, pid} ->
          Enum.each(pids, &(send &1, {:pyctopid, pid})) # update all topics with new pyctopid
          {pid, pids}
        other ->
          IO.puts "Got unexpected message: #{inspect other}"
          {pyctopid, pids}
      after
        5_000 ->
          Pyctopod.write_to_python(pyctopid, :keepalive, :priv_keepalive, 'keepalive')
          {pyctopid, pids}
      end
    loop(pyctopid, pids, testpid)
  end

  defp subscribe(pyctopid, topic) do
    # This function gets spawned every time a new topic is subscribed to.
    # It is responsible for catching any messages that PubSub publishes
    # to its topic and sending those messages to the Python process.
    pyctopid =
      receive do
        {:pyctopid, pid} ->
          pid
        {from, msg} -> 
          Pyctopod.write_to_python(pyctopid, from, topic, msg)
          pyctopid
        other ->
          IO.puts "Got unexpected message in subscription loop: #{inspect other}"
          pyctopid
      end
    subscribe(pyctopid, topic)
  end
end
