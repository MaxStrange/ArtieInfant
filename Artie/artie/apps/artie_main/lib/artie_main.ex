defmodule ArtieMain do
  @moduledoc """
  This is the main module for the Artie Infant application.
  """
  def main(_args) do
    ArtieServer.start_link()
    spin()
  end

  def spin() do
    spin()
  end
end
