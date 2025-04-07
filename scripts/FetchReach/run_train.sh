for STEPS in "50000" "150000" "300000"; do
  for SEED in 42 13 24 18 46 19 28 32 91 12; do 
    react run --env-name FetchReach --saved-model train_SAC --checkpoint $STEPS --seed $SEED 
  done
done


