#!/bin/bash
# dataset_name: facebook ca_GrQc lastfm_asia
# sm: RNS DBS PRBS RES RNES RES_Induction SRW DiffusionTreeS FFS NBRW SBS RW_Starter BFS DFS RW_Jump FrontierS RNNS ShortestPathS CommunitySES CNRWS MHRWS CNARW
# mode: coreball edgeball fireball firecoreball hubball
# FrontierS NBRW RW_Starter RES RNES

# Path to your Python script
python_script="/Users/wangyun/Documents/GitHub/GraphHT/HypothesisTesting/main.py"

#sm=("ours" "RNS" "FFS" "MHRWS" "NBRW" "FrontierS" "RW_Starter" "ShortestPathS" "RES" "RES_Induction" "DBS" "SBS" "SRW")
#sm=("MHRWS" "NBRW" "FrontierS" "RW_Starter" "ShortestPathS" "SRW")
sm=("RES")
#num_samples=10
# Check if the length of sm and sampling_values arrays is the same
#if [ ${#sm[@]} -ne ${#sampling_ratios[@]} ]; then
#  echo "Error: Length of sm and sampling_ratios arrays must be the same."
#  exit 1
#fi

# Loop through the indices of the arrays
for ((i=0; i<${#sm[@]}; i++)); do
  current_sm="${sm[i]}"
#  current_sampling_ratio="${sampling_ratios[i]}"
  echo "Running $python_script with sampling_method: $current_sm"
  nohup python "$python_script" --sampling_method "$current_sm" > trash.log 2>&1 &
done

echo Finished!
