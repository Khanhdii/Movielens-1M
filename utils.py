import torch
from torchmetrics import BLEUScore

def log_loss(logger, loss):
    logger.experiment.add_scalar('Loss/train', loss, global_step=logger.global_step)

def log_bleu_score(logger, bleu_score):
    logger.experiment.add_scalar('BLEU/val', bleu_score, global_step=logger.global_step)

def save_model(model, path):
    torch.save(model.state_dict(), path)
