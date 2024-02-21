#!/bin/bash
# sm: RNS DBS PRBS RES RES_Induction SRW FFS NBRW SBS RW_Starter BFS DFS FrontierS ShortestPathS CommunitySES CNRWS MHRWS

# Path to your main.py
current_folder="$PWD"
python_script="${current_folder}/main.py"

sm=("RES")

for ((i=0; i<${#sm[@]}; i++)); do
  current_sm="${sm[i]}"
  echo "Running $python_script with sampling_method: $current_sm"
  nohup python "$python_script" --sampling_method "$current_sm" > myLog.log 2>&1 &
done

echo Finished!
