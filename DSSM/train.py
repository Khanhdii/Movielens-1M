import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dssm import DSSM
from data import get_data_loaders
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import pandas as pd
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser(description="Train DSSM model on MovieLens 1M dataset")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs to use (0 for CPU, 1 for GPU)")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--log_dir', type=str, default="tb_logs", help="Directory for saving TensorBoard logs")
    parser.add_argument('--binary_classification', type=int, default=1,
                        help="Whether to use binary classification (1) or multi-class classification (0)")
    return parser.parse_args()


class DSSMTrainer(pl.LightningModule):
    def __init__(self, num_users, num_movies, embedding_dim=64, mlp_hidden_size=[256, 128], dropout_prob=0.5,
                 learning_rate=1e-3, binary_classification=False):
        super(DSSMTrainer, self).__init__()
        self.model = DSSM(num_users, num_movies, embedding_dim, mlp_hidden_size, dropout_prob, binary_classification)
        self.learning_rate = learning_rate
        self.binary_classification = binary_classification

        if self.binary_classification:
            print(f"\033[35m{"\nYou are using binary classification.\n"}\033[0m")
            self.loss_fn = nn.BCEWithLogitsLoss()  # Loss for binary classification
        else:
            print(f"\033[35m{"\nYou are using multi-class classification.\n"}\033[0m")
            self.loss_fn = nn.CrossEntropyLoss()  # Loss for multi-class classification (5 classes)

    def forward(self, user, movie):
        return self.model(user, movie)

    def training_step(self, batch, batch_idx):
        user, movie, rating = batch
        output = self.forward(user, movie)

        if self.binary_classification:
            # Binary classification: Output should be 1 value per sample
            output = output.squeeze()  # Remove the extra dimension for binary classification
            loss = self.loss_fn(output, rating.float())  # BCEWithLogitsLoss expects [batch_size]
        else:
            # Multi-class classification: Output should be [batch_size, num_classes]
            output = output.view(-1, 5)  # Ensure output has shape [batch_size, 5]
            loss = self.loss_fn(output, rating)  # CrossEntropyLoss expects [batch_size, num_classes]

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        user, movie, rating = batch
        output = self.forward(user, movie)

        if self.binary_classification:
            # Binary classification: Output should be 1 value per sample
            output = output.squeeze()  # Remove the extra dimension for binary classification
            loss = self.loss_fn(output, rating.float())  # BCEWithLogitsLoss expects [batch_size]
        else:
            # Multi-class classification: Output should be [batch_size, num_classes]
            output = output.view(-1, 5)  # Ensure output has shape [batch_size, 5]
            loss = self.loss_fn(output, rating)  # CrossEntropyLoss expects [batch_size, num_classes]

        return loss

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x for x in outputs]).mean()
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


def main():
    args = parse_args()

    ratings_file = '/mnt/c/Users/Admin/Documents/Recomendation/movielens-1m/ratings.dat'
    movies_file = '/mnt/c/Users/Admin/Documents/Recomendation/movielens-1m/movies.dat'

    train_loader, val_loader = get_data_loaders(ratings_file, movies_file, batch_size=args.batch_size,
                                                binary_classification=args.binary_classification)

    ratings = pd.read_csv(ratings_file, sep='::', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')

    num_users = ratings['userId'].nunique()
    num_movies = ratings['movieId'].nunique()

    model = DSSMTrainer(num_users, num_movies, learning_rate=args.learning_rate,
                        binary_classification=args.binary_classification)

    logger = TensorBoardLogger(args.log_dir, name="dssm_model")

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        verbose=True,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        logger=logger,
        callbacks=[early_stop_callback]
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
