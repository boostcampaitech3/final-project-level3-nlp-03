python train.py --lr 2e-5 \
              --model_name klue/bert-base \
              --num_train_epochs 5 \
              --config_path ./configs/second_gen_bin.yaml \
              --save_steps 500

python train.py --lr 2e-5 \
              --model_name klue/bert-base \
              --num_train_epochs 5 \
              --config_path ./configs/second_korsentence_bin.yaml \
              --save_steps 500

python train.py --lr 2e-5 \
              --model_name klue/bert-base \
              --num_train_epochs 5 \
              --config_path ./configs/second_para_bin.yaml \
              --save_steps 500