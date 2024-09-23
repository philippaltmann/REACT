for SEED in 42 13 24 18 46 19 28 32 91 12; do 
  react baseline1 --env-name FlatGrid11 --saved-model train_ppo --checkpoint 35000 --name FlatGrid11 --seed $SEED --pop-size 10 --iterations 1 --encoding-length 6
done
