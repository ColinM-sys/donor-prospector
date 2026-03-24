#!/bin/bash
pkill -f 'python3 app.py' 2>/dev/null
sleep 1
export PATH="$HOME/.local/bin:$PATH"
cd ~/donor-prospector
nohup python3 app.py > /tmp/donor-prospector.log 2>&1 &
echo "Donor Prospector launched on port 7870 (PID: $!)"
