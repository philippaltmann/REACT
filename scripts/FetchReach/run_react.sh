for S in "50000 50k" "100000 100k" "150000 150k"; do
  set -- $S # $1 -> CHECKPOINT, $2 -> NAME
  for SEED in 42 13 24 18 46 19 28 32 91 12; do 
    react evo --env-name FetchReach --saved-model train_SAC --checkpoint $1 --name Fetch$2 --seed $SEED --pop-size 10 --iterations 40 --encoding-length 9 --is-elitist --crossover 0.75 --mutation 0.5
  done
done
