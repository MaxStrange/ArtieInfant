if ($args.Count -ne 1) {
    Write-Host "Need a name for this node"
    exit 1
}
iex.bat --sname $args.Get(0) -S mix