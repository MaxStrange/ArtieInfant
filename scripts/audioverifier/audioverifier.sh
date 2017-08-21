#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Need a path to search"
    exit 1
fi
find $1 \( -name "*.wav" -o -name "*.WAV" \) -exec sox --info {} \; | grep -e "Sample Rate" -e "Precision" -e "Channels" | awk -F ' : ' '{print $2}'
