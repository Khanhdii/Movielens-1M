import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Model")
    
    # Data parameters
    parser.add_argument('--train_file', type=str, required=True, help="Path to training data file")
    parser.add_argument('--max_len', type=int, default=50, help="Maximum sequence length")
    parser.add_argument('--max_vocab_size', type=int, default=5000, help="Maximum vocabulary size")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=128, help="Dimension of the model")
    parser.add_argument('--nhead', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--dim_feedforward', type=int, default=256, help="Dimension of feedforward network")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate")
    parser.add_argument('--src_vocab_size', type=int, default=5000, help="Source vocabulary size")
    parser.add_argument('--tgt_vocab_size', type=int, default=5000, help="Target vocabulary size")
    
    # Special tokens indices
    parser.add_argument('--src_pad_idx', type=int, default=0, help="Source padding token index")
    parser.add_argument('--tgt_pad_idx', type=int, default=0, help="Target padding token index")
    parser.add_argument('--src_sos_idx', type=int, default=1, help="Source start of sequence token index")
    parser.add_argument('--tgt_sos_idx', type=int, default=1, help="Target start of sequence token index")
    parser.add_argument('--src_eos_idx', type=int, default=2, help="Source end of sequence token index")
    parser.add_argument('--tgt_eos_idx', type=int, default=2, help="Target end of sequence token index")
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (cuda/cpu)")
    
    return parser.parse_args()
