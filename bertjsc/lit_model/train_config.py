import torch
import pytorch_lightning as pl

class TrainConfig(pl.LightningModule):
    
    def training_step(self, batch, batch_idx):
        output = self(**batch)
        
        loss = output.loss
        self.log('train_loss', loss, on_step=True)
        return {"loss": loss}
        
    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        
        loss = output.loss
        self.log("val_loss", loss, on_epoch=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-6)
        return optimizer