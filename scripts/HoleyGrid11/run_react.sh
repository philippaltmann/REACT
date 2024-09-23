for SEED in 42 13 24 18 46 19 28 32 91 12; do 
  react evo --env-name HoleyGrid11 --saved-model train_ppo --checkpoint 150000 --name HoleyGrid11 --seed $SEED --pop-size 10 --iterations 40 --encoding-length 6 --plot-frequency 10 --is-elitist --crossover 0.75 --mutation 0.5
done
