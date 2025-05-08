import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from model.encoder import EncoderLayer
from model.decoder import DecoderLayer
from torchmetrics.text import BLEUScore
from torchmetrics import Perplexity

class TransformerModel(pl.LightningModule):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.save_hyperparameters()
        
        # Save special token indices
        self.src_pad_idx = args.src_pad_idx
        self.tgt_pad_idx = args.tgt_pad_idx
        self.src_sos_idx = args.src_sos_idx
        self.tgt_sos_idx = args.tgt_sos_idx
        self.src_eos_idx = args.src_eos_idx
        self.tgt_eos_idx = args.tgt_eos_idx
        
        # Embedding layers
        self.src_embedding = nn.Embedding(args.src_vocab_size, args.d_model, padding_idx=self.src_pad_idx)
        self.tgt_embedding = nn.Embedding(args.tgt_vocab_size, args.d_model, padding_idx=self.tgt_pad_idx)
        
        # Transformer layers
        self.encoder = EncoderLayer(d_model=args.d_model, 
                                  nhead=args.nhead, 
                                  dim_feedforward=args.dim_feedforward, 
                                  dropout=args.dropout)

        self.decoder = DecoderLayer(d_model=args.d_model, 
                                  nhead=args.nhead, 
                                  dim_feedforward=args.dim_feedforward, 
                                  dropout=args.dropout)

        self.fc_out = nn.Linear(args.d_model, args.tgt_vocab_size)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_pad_idx)
        
        # Metrics
        self.val_bleu = BLEUScore()

    def forward(self, src, tgt):
        # Embedding
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        
        # Scale embeddings
        src = src * math.sqrt(self.hparams.args.d_model)
        tgt = tgt * math.sqrt(self.hparams.args.d_model)
        
        # Transformer layers
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt, encoder_output)
        
        # Output layer
        output = self.fc_out(decoder_output)
        return output

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt[:, :-1])
        
        # Calculate loss
        loss = self.criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt[:, :-1])
        
        # Calculate loss
        loss = self.criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        
        # # Calculate metrics
        # preds = torch.argmax(output, dim=-1)
        
        # # Chuyển đổi predictions và targets thành text
        # pred_texts = []
        # tgt_texts = []
        
        # for pred, tgt_seq in zip(preds, tgt[:, 1:]):
        #     # Chuyển indices thành text
        #     pred_text = ' '.join([str(idx.item()) for idx in pred])
        #     tgt_text = ' '.join([str(idx.item()) for idx in tgt_seq])
            
        #     pred_texts.append(pred_text)
        #     tgt_texts.append([tgt_text])  # BLEU cần list of references
        
        # # Lưu lại để tính BLEU vào cuối training
        # self.val_pred_texts = getattr(self, 'val_pred_texts', []) + pred_texts
        # self.val_tgt_texts = getattr(self, 'val_tgt_texts', []) + tgt_texts
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        
        return loss

    def on_fit_end(self):
        # Tính BLEU score cho toàn bộ validation set khi kết thúc training
        if hasattr(self, 'val_pred_texts') and hasattr(self, 'val_tgt_texts'):
            bleu_score = self.val_bleu(self.val_pred_texts, self.val_tgt_texts)
            print(f"\nFinal BLEU Score: {bleu_score:.4f}")
            
            # In một số mẫu dự đoán
            if len(self.val_pred_texts) > 0:
                print("\nSample predictions:")
                for i in range(min(3, len(self.val_pred_texts))):
                    print(f"Pred: {self.val_pred_texts[i]}")
                    print(f"Target: {self.val_tgt_texts[i][0]}")
            
            # Reset lists
            self.val_pred_texts = []
            self.val_tgt_texts = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        # Thêm Early Stopping
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min',
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            },
            'callbacks': [early_stop_callback]
        }
    

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src, tgt[:, :-1])

        # Get predicted tokens (argmax to choose most likely token)
        preds = torch.argmax(output, dim=-1)

        # Convert predictions and targets to text (list of tokens)
        pred_texts = []
        tgt_texts = []

        for pred, tgt_seq in zip(preds, tgt[:, 1:]):
            pred_text = ' '.join([str(idx.item()) for idx in pred])
            tgt_text = ' '.join([str(idx.item()) for idx in tgt_seq])

            pred_texts.append(pred_text)
            tgt_texts.append([tgt_text])  # BLEU requires a list of references

        # Store predictions and targets for later evaluation
        self.test_pred_texts = getattr(self, 'test_pred_texts', []) + pred_texts
        self.test_tgt_texts = getattr(self, 'test_tgt_texts', []) + tgt_texts

        return {"test_loss": self.criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))}


    def on_test_end(self):
        print("Testing finished, calculating BLEU score...")
        if hasattr(self, 'test_pred_texts') and hasattr(self, 'test_tgt_texts'):
            bleu_score = self.val_bleu(self.test_pred_texts, self.test_tgt_texts)
            print(f"\nFinal BLEU Score on Test Set: {bleu_score:.4f}")

            # Reset lists
            self.test_pred_texts = []
            self.test_tgt_texts = []
