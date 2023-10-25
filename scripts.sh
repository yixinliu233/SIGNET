# with ground-truth explain
python main.py --dataset mnist0 --epoch 50 --lr 0.01 --explainer_layers 2 --explainer_hidden_dim 16 --explainer_model mlp --encoder_layers 2 --batch_size_test 64
python main.py --dataset mnist1 --epoch 50 --lr 0.01 --explainer_layers 2 --explainer_hidden_dim 16 --explainer_model mlp --encoder_layers 2 --batch_size_test 64
python main.py --dataset mutag --epoch 50 --lr 0.01 --explainer_layers 5 --explainer_hidden_dim 4
# without ground-truth explain
python main.py --dataset AIDS --epoch 1000 --lr 0.0001 --hidden_dim 16
python main.py --dataset DHFR --epoch 1000 --lr 0.0001 --hidden_dim 128
python main.py --dataset BZR --epoch 1000 --lr 0.0001 --hidden_dim 128
python main.py --dataset COX2 --epoch 1000 --lr 0.0001 --hidden_dim 64
