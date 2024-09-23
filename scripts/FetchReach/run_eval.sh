for S in "100000 100k" "3000000 3M" "5000000 5M"; do
  set -- $S # $1 -> STEPS, $2 -> STEPS_NAME
  react fidelity Fetch$2
  # for SEED in 42 13 24 18 46 19 28 32 91 12; do 
  #   react eval --env-name FetchReach --exp-name Fetch$2-$SEED --saved-model train1_sac --checkpoint $1 --seed 42 --render
  # done
done
