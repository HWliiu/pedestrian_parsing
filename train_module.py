import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import draw_segmentation_masks, save_image

from bisenetv2 import BiSeNetV2


class OhemCELoss(nn.Module):
    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(
            torch.tensor(thresh, requires_grad=False, dtype=torch.float)
        )
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction="none")

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class Evaluator:
    def __init__(self) -> None:
        self.reset()

    @staticmethod
    def fast_hist(a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

    @staticmethod
    def miou(logits, labels, numClass):
        hist = np.zeros((numClass, numClass))
        for logit, label in zip(logits, labels):
            hist += Evaluator.fast_hist(label.flatten(), logit.flatten(), numClass)

        miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        return np.mean(miou)

    def reset(self):
        self.total_miou = 0.0
        self.batch_count = 0

    def update(self, logits, labels):
        self.total_miou += Evaluator.miou(
            logits=logits.argmax(axis=1), labels=labels, numClass=logits.shape[1]
        )
        self.batch_count += 1

    def calculate_result(self):
        result = self.total_miou / self.batch_count
        self.reset()
        return result


class PedestrianParsing(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.bisenetv2 = BiSeNetV2(n_classes=8)
        num_aux_heads = 4
        self.criteria_pre = OhemCELoss(0.7)
        self.criteria_aux = [OhemCELoss(0.7) for _ in range(num_aux_heads)]
        self.evaluator = Evaluator()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        images, labels = batch

        logits, *logits_aux = self.bisenetv2(images)
        loss_pre = self.criteria_pre(logits, labels.long())
        loss_aux = [
            crit(lgt, labels.long()) for crit, lgt in zip(self.criteria_aux, logits_aux)
        ]
        loss = loss_pre + sum(loss_aux)

        self.log("loss", loss, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        images, labels = batch
        logits = self.bisenetv2.forward(images)[0]
        self.evaluator.update(logits.cpu().numpy(), labels.cpu().numpy())

    def on_validation_epoch_end(self):
        miou = self.evaluator.calculate_result()
        self.log("miou", miou, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        images, _, _ = batch

        logits = self.bisenetv2(
            TF.normalize(
                images,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
        preds = torch.argmax(logits[0], dim=1)
        preds = nn.functional.one_hot(preds).bool().permute(0, 3, 1, 2)
        imgs_vis = []
        colors = [
            "#CCCCCC",
            "#666699",
            "#FFCC33",
            "#FF6666",
            "#99CC00",
            "#FF9900",
            "#CCCC33",
            "#0099CC",
        ]
        for img, labels in zip(images.cpu(), preds.cpu()):
            imgs_vis.append(
                draw_segmentation_masks(
                    (img * 255).to(torch.uint8), labels, alpha=0.5, colors=colors
                )
            )
        save_image(torch.stack(imgs_vis) / 255.0, f"lightning_logs/{batch_idx:0>2}.png")

    def configure_optimizers(self):
        optimizer = optim.RAdam(
            self.bisenetv2.parameters(),
            lr=1e-3,
        )
        lr_scheduler_config = {
            "scheduler": StepLR(optimizer, step_size=50, gamma=0.5),
        }
        return [optimizer], [lr_scheduler_config]
