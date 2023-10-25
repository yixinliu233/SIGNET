import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='SIGNET')
    parser.add_argument('--dataset', type=str, default='mutag')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=9999)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--encoder_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--pooling', type=str, default='add', choices=['add', 'max'])
    parser.add_argument('--readout', type=str, default='concat', choices=['concat', 'add', 'last'])
    parser.add_argument('--explainer_model', type=str, default='gin', choices=['mlp', 'gin'])
    parser.add_argument('--explainer_layers', type=int, default=5)
    parser.add_argument('--explainer_hidden_dim', type=int, default=8)
    parser.add_argument('--explainer_readout', type=str, default='add', choices=['concat', 'add', 'last'])

    return parser.parse_args()