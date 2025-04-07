for SEED in 42 13 24 18 46 19 28 32 91 12; do 
  react run --env-name HoleyGrid11 --saved-model train_PPO --checkpoint 150000 --seed $SEED
done
