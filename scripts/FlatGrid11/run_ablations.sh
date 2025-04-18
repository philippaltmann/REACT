BASE="--env-name FlatGrid11 --saved-model train_PPO --checkpoint 35000 --pop-size 10 --iterations 40 --encoding-length 6 --plot-frequency 40 --is-elitist --crossover 0.75 --mutation 0.5" 

for SEED in 42 13 24 18 46 19 28 32 91 12; do 
  # react evo --env-name FlatGrid11 --saved-model train_PPO --checkpoint 35000 --name FlatGrid11 --seed $SEED --pop-size 10 --iterations 40 --encoding-length 6 --plot-frequency 10 --is-elitist --crossover 0.75 --mutation 0.5
  echo "Running $SEED"
  react evo $BASE --name FlatGrid11 --seed $SEED --w1 1 --w2 0 --w3 0 --w4 0 
  react evo $BASE --name FlatGrid11 --seed $SEED --w1 0 --w2 1 --w3 1 --w4 1
  react evo $BASE --name FlatGrid11 --seed $SEED --w1 1 --w2 1 --w3 1 --w4 0
  react evo $BASE --name FlatGrid11 --seed $SEED --w1 0 --w2 1 --w3 0 --w4 0
  react evo $BASE --name FlatGrid11 --seed $SEED --w1 0 --w2 0 --w3 1 --w4 0
  react evo $BASE --name FlatGrid11 --seed $SEED --w1 0 --w2 0 --w3 0 --w4 0
done
