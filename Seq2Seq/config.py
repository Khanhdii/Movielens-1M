import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--use_beam_in_train', action='store_true', help='Log BLEU with Beam Search during training')
    return parser.parse_args()
