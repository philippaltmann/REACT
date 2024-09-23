for S in "100000 100k" "3000000 3M" "5000000 5M"; do
  set -- $S # $1 -> STEPS, $2 -> STEPS_NAME
  for SEED in 42 13 24 18 46 19 28 32 91 12; do 
    react baseline1 --env-name FetchReach --saved-model train_sac --checkpoint $1 --name Fetch$2 --seed $SEED --pop-size 30 --iterations 1 --encoding-length 9
  done
done
