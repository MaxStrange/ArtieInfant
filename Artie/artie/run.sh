if [ "$#" -ne 1 ]; then
    echo "Need a name for this node"
    exit 1
fi

iex --sname $1 -S mix
