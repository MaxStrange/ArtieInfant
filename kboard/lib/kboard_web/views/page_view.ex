defmodule KboardWeb.PageView do
  use KboardWeb, :view

  def handler_images(name) do
    case name do
      "accuracy" -> "/images/acc.png"
      "fscore"   -> "/images/fscore.png"
      "loss"     -> "/images/loss.png"
    end
  end
end
