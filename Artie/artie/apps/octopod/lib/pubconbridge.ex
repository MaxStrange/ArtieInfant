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
  def start(pyctopid, testpid \\ nil, eavesdropper_pid \\ nil) do
    testpid = if (testpid == nil), do: self(), else: testpid
    pid = spawn fn -> loop(pyctopid, [], testpid, eavesdropper_pid) end
    {:ok, pid}
  end

  defp loop(pyctopid, pids, testpid, eavesdropper_pid) do
    {pyctopid, pids} =
      receive do
        {from, :test, msg} ->
          handle_clause({from, :test, msg}, testpid, eavesdropper_pid)
          {pyctopid, pids}
        {from, topic, msg} ->
          handle_clause({from, topic, msg}, eavesdropper_pid)
          {pyctopid, pids}
        {:subscribe, topic} ->
          pid = handle_clause({:subscribe, topic}, pyctopid, eavesdropper_pid)
          {pyctopid, [pid | pids]}
        {:pyctopid, pid} ->
          handle_clause({:pyctopid, pid}, pids, eavesdropper_pid)
          {pid, pids}
        other ->
          handle_clause(other)
          {pyctopid, pids}
      after
        5_000 ->
          Pyctopod.write_to_python(pyctopid, :keepalive, :priv_keepalive, 'keepalive')
          {pyctopid, pids}
      end
    loop(pyctopid, pids, testpid, eavesdropper_pid)
  end

  defp handle_clause({from, :test, msg}, testpid, eavesdropper_pid) do
    ## Outdated test clause - superceded by the eavesdropper

    # For testing purposes - we listen to the test topic on our test process
    send(testpid, {from, :test, msg})

    # Send the eavesdropper a msg
    if (eavesdropper_pid != nil), do: send eavesdropper_pid, {from, :test, msg}

    # Since :test is a perfectly valid topic name, we will also pass it on
    PubSub.publish(:test, {from, msg})
  end

  defp handle_clause({:subscribe, topic}, pyctopid, eavesdropper_pid) do
    ## Alert the PubSub module that our python instance wants to subscribe to topic

    # Eavesdrop
    if (eavesdropper_pid != nil), do: send eavesdropper_pid, {:subscribe, topic}

    pid = spawn_link fn -> subscribe(pyctopid, topic) end
    PubSub.subscribe(pid, topic)

    pid
  end

  defp handle_clause({:pyctopid, pid}, pids, eavesdropper_pid) do
    ## Update the pid for python communication

    # Eavesdrop
    if (eavesdropper_pid != nil), do: send eavesdropper_pid, {:pyctopid, pid}

    Enum.each(pids, &(send &1, {:pyctopid, pid})) # update all topics with new pyctopid
  end

  defp handle_clause({from, topic, msg}, eavesdropper_pid) do
    ## This is the workhorse clause - it publishes the given msg to the given topic

    # Eavesdrop
    if (eavesdropper_pid != nil), do: send eavesdropper_pid, {from, topic, msg}

    PubSub.publish(topic, {from, msg})
  end

  defp handle_clause(other) do
    ## Catchall

    IO.puts "Got unexpected message: #{inspect other}"
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
