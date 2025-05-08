import pytorch_lightning as pl
import torch
from torchmetrics import BLEUScore
from utils import log_loss, log_bleu_score, save_model
from torch.optim import Adam

class Trainer(pl.LightningModule):
    def __init__(self, args, model, train_loader, val_loader, test_loader):
        super(Trainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.bleu_metric = BLEUScore()

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        output = self.model(src, tgt)
        loss = self.model.calculate_loss(output, tgt)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        output = self.model(src, tgt)
        loss = self.model.calculate_loss(output, tgt)
        bleu = self.bleu_metric(output, tgt)
        return {'val_loss': loss, 'val_bleu': bleu}

    def on_validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_bleu = torch.stack([x['val_bleu'] for x in outputs]).mean()
        log_loss(self.logger, avg_loss)
        log_bleu_score(self.logger, avg_bleu)
        return {'val_loss': avg_loss, 'val_bleu': avg_bleu}

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        output = self.model(src, tgt)
        loss = self.model.calculate_loss(output, tgt)
        bleu = self.bleu_metric(output, tgt)
        return {'test_loss': loss, 'test_bleu': bleu}

    def on_test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_bleu = torch.stack([x['test_bleu'] for x in outputs]).mean()
        log_loss(self.logger, avg_loss)
        log_bleu_score(self.logger, avg_bleu)
        return {'test_loss': avg_loss, 'test_bleu': avg_bleu}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def train(self):
        trainer = pl.Trainer(max_epochs=self.args.epochs, 
                             devices=1, 
                             accelerator="gpu")

        trainer.fit(self.model, self.train_loader, self.val_loader)
        
        # Evaluate on the test set after training
        trainer.test(self.model, self.test_loader)
