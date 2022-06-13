python train_nli.py --lr 2e-5 \
              --model_name klue/bert-base \
              --num_train_epochs 1 \
              --config_path ./configs/first_korNLI.yaml \
              --train_bs 32 \
              --save_steps 1000