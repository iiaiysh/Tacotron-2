#!/bin/sh
source deactivate
conda activate rayhane-ysh
dir1=$(find . -name '*tacotron_events*')
echo "$dir1"
echo ""


port=7006

for dir in $dir1
do
    events=$(find $dir -name 'events.out*')
    if [ "$events" = "" ]; then
    # $events is empty
        echo "$dir is empty"
        continue
    fi

    tensorboard --logdir $dir --port $port &
    echo "$dir is running at $port"

    port=$((port+1))

done

