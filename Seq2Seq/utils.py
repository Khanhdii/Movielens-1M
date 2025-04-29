import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

def read_text(file_name: str) -> pd.DataFrame:
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
    lines = [line for line in lines if len(line) == 3]
    return pd.DataFrame(lines, columns=['english', 'french', 'attribution'])

def build_vocab(sentences, tokenizer, specials=['<unk>', '<pad>', '<sos>', '<eos>']):
    vocab = build_vocab_from_iterator(map(tokenizer, sentences), specials=specials)
    vocab.set_default_index(vocab['<unk>'])
    return vocab

class TranslationDataset(Dataset):
    def __init__(self, df, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer, max_len=30):
        self.df = df
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab
        self.en_tokenizer = en_tokenizer
        self.fr_tokenizer = fr_tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        en_tokens = ['<sos>'] + self.en_tokenizer(self.df.iloc[idx]['english']) + ['<eos>']
        fr_tokens = ['<sos>'] + self.fr_tokenizer(self.df.iloc[idx]['french']) + ['<eos>']
        en_indices = [self.en_vocab[t] for t in en_tokens][:self.max_len]
        fr_indices = [self.fr_vocab[t] for t in fr_tokens][:self.max_len]
        en_indices += [self.en_vocab['<pad>']] * (self.max_len - len(en_indices))
        fr_indices += [self.fr_vocab['<pad>']] * (self.max_len - len(fr_indices))
        return {'english': torch.tensor(en_indices), 'french': torch.tensor(fr_indices)}

def get_dataloaders(file_path, batch_size, max_len=30):
    df = read_text(file_path)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)

    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

    en_vocab = build_vocab(train_df['english'], en_tokenizer)
    fr_vocab = build_vocab(train_df['french'], fr_tokenizer)

    train_ds = TranslationDataset(train_df, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer, max_len)
    val_ds   = TranslationDataset(val_df, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer, max_len)
    test_ds  = TranslationDataset(test_df, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader, en_vocab, fr_vocab

def load_glove_embeddings(vocab, embedding_dim=300):
    try:
        glove = GloVe(name='840B', dim=embedding_dim)
        matrix = torch.zeros(len(vocab), embedding_dim)
        for i, token in enumerate(vocab.get_itos()):
            matrix[i] = glove[token] if token in glove.stoi else torch.randn(embedding_dim)
        return matrix
    except:
        print("Warning: Cannot load GloVe. Using random embeddings instead.")
        return torch.randn(len(vocab), embedding_dim)

def get_logger(log_dir):
    return TensorBoardLogger(save_dir=log_dir, name="seq2seq_attention")
