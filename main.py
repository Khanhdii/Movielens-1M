import torch
import pytorch_lightning as pl
from config import parse_args
from data import get_data_loaders
from model.transformer import TransformerModel
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    pl.seed_everything(42)
    
    # Enable Tensor Cores for better performance
    torch.set_float32_matmul_precision('high')
    
    # Parse arguments
    args = parse_args()
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(args)
    
    # Create model
    model = TransformerModel(args)
    
    # Create TensorBoard logger
    logger = TensorBoardLogger("logs", name="transformer_model")
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='transformer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    # Create trainer with optimized settings for large batch size
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',  # Sử dụng mixed precision để tiết kiệm VRAM
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,  # Không cần accumulate vì batch size đã lớn
        log_every_n_steps=50,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        deterministic=False,
        benchmark=True,
        num_sanity_val_steps=0
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()
