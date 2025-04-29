import torch
import torch.nn as nn
import pytorch_lightning as pl
from config import get_args
from utils import get_logger, get_dataloaders, load_glove_embeddings
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def main():
    args = get_args()

    train_loader, val_loader, test_loader, en_vocab, fr_vocab = get_dataloaders(
        'Data/fra.txt', batch_size=args.batch_size, max_len=args.max_len
    )

    en_embedding_matrix = load_glove_embeddings(en_vocab)
    fr_embedding_matrix = load_glove_embeddings(fr_vocab)

    en_embedding = nn.Embedding.from_pretrained(en_embedding_matrix, freeze=False)
    fr_embedding = nn.Embedding.from_pretrained(fr_embedding_matrix, freeze=False)

    encoder = Encoder(en_embedding, hidden_size=64, num_heads=args.num_heads)
    decoder = Decoder(fr_embedding, hidden_size=64, output_size=len(fr_vocab), num_heads=args.num_heads)

    model = Seq2Seq(encoder, decoder, fr_vocab['<sos>'], fr_vocab['<pad>'], args.max_len, args.lr)
    model.use_beam_in_train = args.use_beam_in_train

    logger = get_logger(args.log_dir)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints/', save_top_k=1, mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=20, mode='min')

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="cuda"
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)

if __name__ == "__main__":
    import nltk
    nltk.download('punkt')
    main()
