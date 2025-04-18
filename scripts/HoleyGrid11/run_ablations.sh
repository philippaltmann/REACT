BASE="--env-name HoleyGrid11 --saved-model train_PPO --checkpoint 150000 --pop-size 10 --iterations 40 --encoding-length 6 --is-elitist --crossover 0.75 --mutation 0.5" 

for SEED in 42 13 24 18 46 19 28 32 91 12; do 
  echo "Running $SEED"
  react evo $BASE --name HoleyGrid11 --seed $SEED --w1 1 --w2 0 --w3 0 --w4 0
  # react evo $BASE --name HoleyGrid11 --seed $SEED --w1 0 --w2 1 --w3 1 --w4 1
  react evo $BASE --name HoleyGrid11 --seed $SEED --w1 1 --w2 1 --w3 1 --w4 0
  react evo $BASE --name HoleyGrid11 --seed $SEED --w1 0 --w2 1 --w3 0 --w4 0
  react evo $BASE --name HoleyGrid11 --seed $SEED --w1 0 --w2 0 --w3 1 --w4 0
  react evo $BASE --name HoleyGrid11 --seed $SEED --w1 0 --w2 0 --w3 0 --w4 0
  
done
