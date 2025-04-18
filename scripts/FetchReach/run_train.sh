for S in "50000 50k" "100000 100k" "200000 200k"; do
  set -- $S # $1 -> CHECKPOINT, $2 -> NAME
  for SEED in 42 13 24 18 46 19 28 32 91 12; do 
    react run --env-name FetchReach --saved-model train_SAC --checkpoint $1 --name Fetch$2 --seed $SEED 
  done
done


