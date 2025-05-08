import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self, file_path, max_len=50, max_vocab_size=5000):
        self.pairs = self.load_data(file_path)
        self.max_len = max_len
        self.max_vocab_size = max_vocab_size
        self.src_vocab, self.tgt_vocab = self.build_vocab()
        # Tạo index_to_token cho ngôn ngữ đích
        self.index_to_token = {idx: token for token, idx in self.tgt_vocab.items()}

    def load_data(self, file_path):
        pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                src, tgt, _ = line.strip().split('\t')
                pairs.append((src, tgt))
        return pairs

    def build_vocab(self):
        src_counter = Counter()
        tgt_counter = Counter()

        for src, tgt in self.pairs:
            src_counter.update(src.split())
            tgt_counter.update(tgt.split())

        src_vocab = {word: idx+2 for idx, (word, _) in enumerate(src_counter.most_common(self.max_vocab_size-3))}
        tgt_vocab = {word: idx+2 for idx, (word, _) in enumerate(tgt_counter.most_common(self.max_vocab_size-3))}

        src_vocab['<PAD>'] = 0
        src_vocab['<SOS>'] = 1
        src_vocab['<EOS>'] = 2
        tgt_vocab['<PAD>'] = 0
        tgt_vocab['<SOS>'] = 1
        tgt_vocab['<EOS>'] = 2

        return src_vocab, tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        
        src_tokens = ['<SOS>'] + src.split() + ['<EOS>']
        tgt_tokens = ['<SOS>'] + tgt.split() + ['<EOS>']
        
        src_indices = [self.src_vocab.get(token, 0) for token in src_tokens]
        tgt_indices = [self.tgt_vocab.get(token, 0) for token in tgt_tokens]
        
        src_indices = src_indices[:self.max_len-1] + [0] * (self.max_len - len(src_indices))
        tgt_indices = tgt_indices[:self.max_len-1] + [0] * (self.max_len - len(tgt_indices))
        
        src_tensor = torch.tensor(src_indices)
        tgt_tensor = torch.tensor(tgt_indices)
        
        return src_tensor, tgt_tensor
def collate_fn(batch):
    # Padding các chuỗi về cùng một độ dài
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_batch, tgt_batch

def get_data_loaders(args):
    # Load dataset
    full_dataset = TranslationDataset(args.train_file, max_len=args.max_len, max_vocab_size=args.max_vocab_size)

    # Lấy index_to_token từ dataset
    args.index_to_token = full_dataset.index_to_token

    # Calculate lengths for train, validation, test splits
    train_len = int(0.8 * len(full_dataset))
    val_len = int(0.1 * len(full_dataset))
    test_len = len(full_dataset) - train_len - val_len

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

    return train_loader, val_loader, test_loader