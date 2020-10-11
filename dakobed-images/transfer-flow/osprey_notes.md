ssh -Y -L 16006:127.0.0.1:6006 darrd@osprey.ocean.washington.edu


# Start a tmux session

tmux new -s tensorflow_session

## Create a new pane
control + b %
## Navigate to new pane
control + b ->

## Run tensorboard in one of the panes
tensorboard --logdir=logs 


## tmux cheat sheat

https://tmuxcheatsheet.com/
