import pytorch_lightning as pl

from data_module import DataModule
from train_module import PedestrianParsing


def main():
    data = DataModule()
    data.setup()
    model = PedestrianParsing()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        benchmark=True,
        max_epochs=600,
        precision=16,
        num_sanity_val_steps=0,
        enable_model_summary=True,
        check_val_every_n_epoch=20,
        log_every_n_steps=10,
    )
    trainer.fit(
        model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
    trainer.test(
        model,
        dataloaders=data.test_dataloader(),
        # ckpt_path="lightning_logs/version_0/checkpoints/epoch=599-step=24600.ckpt",
    )


if __name__ == "__main__":
    main()
