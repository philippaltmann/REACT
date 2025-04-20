BASE="--env-name FetchReach --saved-model train_SAC --pop-size 10 --encoding-length 9"
REACT="--iterations 40 --is-elitist --crossover 0.75 --mutation 0.5"

for S in "50000 50k" "100000 100k" "150000 150k"; do
  set -- $S # $1 -> CHECKPOINT, $2 -> NAME
  for SEED in 42 13 24 18 46 19 28 32 91 12; do 
    echo "Running $SEED"
    react evo $BASE $REACT --checkpoint $1 --name Fetch$2 --seed $SEED --w1 1 --w2 0 --w3 0 --w4 0  # REACT_G
    # react evo $BASE $REACT --checkpoint $1 --name Fetch$2 --seed $SEED --w1 0 --w2 1 --w3 1 --w4 1  # REACT_D
    react evo $BASE $REACT --checkpoint $1 --name Fetch$2 --seed $SEED --w1 1 --w2 1 --w3 1 --w4 0  # REACT P
    react evo $BASE $REACT --checkpoint $1 --name Fetch$2 --seed $SEED --w1 0 --w2 1 --w3 0 --w4 0  # REACT_L
    react evo $BASE $REACT --checkpoint $1 --name Fetch$2 --seed $SEED --w1 0 --w2 0 --w3 1 --w4 0  # REACT_C
    react evo $BASE $REACT --checkpoint $1 --name Fetch$2 --seed $SEED --w1 0 --w2 0 --w3 0 --w4 0  # REACT_F
    react evo $BASE --checkpoint $1 --name Fetch$2  --seed $SEED --iterations 0  # Random 
  done
done
