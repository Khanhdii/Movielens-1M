import torch
import torch.nn as nn
import pytorch_lightning as pl
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class Seq2Seq(pl.LightningModule):
    def __init__(self, encoder, decoder, sos_token, pad_token, max_len, lr):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_token = sos_token
        self.pad_token = pad_token
        self.max_len = max_len
        self.lr = lr

    def reshape_bidirectional(self, h):
        return (h[0:2] + h[2:4]) / 2

    def beam_search_decode(self, encoder_outputs, hidden, cell, beam_width=3):
        batch_size = encoder_outputs.size(0)
        sequences = [[torch.full((batch_size, 1), self.sos_token).to(self.device), 0.0, hidden, cell]]

        for _ in range(self.max_len):
            all_candidates = []
            for seq, score, h, c in sequences:
                output, h_new, c_new = self.decoder(seq[:, -1:], encoder_outputs, h, c)
                probs = nn.functional.log_softmax(output[:, -1, :], dim=-1)
                topk_probs, topk_idx = probs.topk(beam_width)

                for i in range(beam_width):
                    candidate_seq = torch.cat([seq, topk_idx[:, i].unsqueeze(1)], dim=1)
                    candidate_score = score + topk_probs[:, i].sum().item()
                    all_candidates.append([candidate_seq, candidate_score, h_new, c_new])

            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences = ordered[:beam_width]

        best_seq = sequences[0][0][:, 1:]  # Bỏ SOS token
        return best_seq

    def forward(self, src, use_beam=False):
        encoder_outputs, (hidden, cell) = self.encoder(src)
        hidden = self.reshape_bidirectional(hidden)
        cell = self.reshape_bidirectional(cell)

        if use_beam:
            decoded_seq = self.beam_search_decode(encoder_outputs, hidden, cell, beam_width=3)
            return decoded_seq.unsqueeze(1)
        else:
            batch_size = src.size(0)
            decoder_input = torch.full((batch_size, 1), self.sos_token).to(self.device)
            outputs = []

            for _ in range(self.max_len):
                output, hidden, cell = self.decoder(decoder_input, encoder_outputs, hidden, cell)
                outputs.append(output)
                decoder_input = output.argmax(-1)

            outputs = torch.cat(outputs, dim=1)
            return outputs  # Trả về logits

    def compute_bleu(self, preds, targets):
        total_bleu = 0
        preds = preds.cpu().tolist()
        targets = targets.cpu().tolist()
        smoothie = SmoothingFunction().method4
        for pred_seq, target_seq in zip(preds, targets):
            pred_tokens = [str(tok) for tok in pred_seq if tok != self.pad_token]
            target_tokens = [[str(tok) for tok in target_seq if tok != self.pad_token]]
            bleu = sentence_bleu(target_tokens, pred_tokens, weights=(0.5, 0.5), smoothing_function=smoothie)
            total_bleu += bleu
        return total_bleu / len(preds)

    def common_step(self, batch, stage):
        src, tgt = batch['english'], batch['french']

        output = self(src, use_beam=False)
        loss = nn.CrossEntropyLoss(ignore_index=self.pad_token)(
            output.view(-1, output.size(-1)),
            tgt.view(-1)
        )

        # Log BLEU nếu là val/test hoặc bật chế độ beam khi train
        if stage != "train" or (hasattr(self, 'use_beam_in_train') and self.use_beam_in_train):
            pred_seq = self(src, use_beam=True)
            preds = pred_seq.squeeze(1)
            bleu_score = self.compute_bleu(preds, tgt)
            self.log(f"{stage}_bleu", bleu_score, prog_bar=True)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
