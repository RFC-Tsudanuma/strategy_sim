#!/bin/bash
dir=`ls -1 ./log/ | sort | tail -1`
uv run play.py --checkpoint ./log/$dir/checkpoints/best_agent.pt 
