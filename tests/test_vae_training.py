"""Regression tests for reconstruction scaling and post-warmup selection."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bin.model import ConditionalVAE, VAELoss, train_vae_with_test_set as legacy_train
from bin import train_utils

class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
    def forward(self, x, labels=None):
        mu = self.weight.expand(len(x), 2)
        return self.weight.expand_as(x), x, mu, torch.zeros_like(mu), mu, None

class ScheduledLoss(nn.Module):
    # Two calls per epoch: training and validation.
    values = [0.0, 1.0, 5.0, 4.0, 6.0]
    def __init__(self):
        super().__init__()
        self.beta = 0.0
        self.calls = 0
    def forward(self, pred, x, mu, logvar):
        value = self.values[self.calls // 2]
        self.calls += 1
        return pred.sum() * 0 + value, {'recon': value, 'kl': 0.0}

class VAETrainingTests(unittest.TestCase):
    def test_loss_scale_and_gradients(self):
        x = torch.zeros(2, 3)
        pred = torch.ones_like(x, requires_grad=True)
        mu = torch.ones(2, 2)
        lv = torch.zeros_like(mu)
        criterion = VAELoss(beta=0.5)
        loss, metrics = criterion(pred, x, mu, lv)
        self.assertEqual(metrics, {'recon': 3.0, 'kl': 1.0})
        self.assertAlmostEqual(loss.item(), 3.5)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.ones_like(pred))
        replicated, _ = criterion(pred.repeat(2, 1), x.repeat(2, 1), mu.repeat(2, 1), lv.repeat(2, 1))
        self.assertEqual(replicated.item(), loss.item())
        more_genes, _ = criterion(pred.repeat(1, 2), x.repeat(1, 2), mu, lv)
        self.assertEqual(more_genes.item(), 6.5)
        reconstruction_only, _ = criterion(pred, x, mu, lv, training=False)
        self.assertEqual(reconstruction_only.item(), 3.0)

    def test_selection_scheduler_and_patience_start_after_warmup(self):
        loader = DataLoader(torch.ones(2, 3), batch_size=2)
        with tempfile.TemporaryDirectory() as tmp, patch.object(train_utils, 'VAELoss', ScheduledLoss), patch.object(train_utils.torch.optim.lr_scheduler, 'ReduceLROnPlateau') as scheduler, patch.object(train_utils.torch, 'save', wraps=torch.save) as save:
            history = train_utils.train_vae_with_test_set(TinyVAE(), loader, loader,
                n_epochs=5, warmup_epochs=2, early_stopping_patience=1,
                learning_rate=0, device='cpu', ckpt_path=str(Path(tmp) / 'best.pth'))
            self.assertEqual(history['test_loss'], ScheduledLoss.values)
            self.assertEqual(save.call_count, 2)
            self.assertEqual([c.args[0] for c in scheduler.return_value.step.call_args_list], [5, 4, 6])

    def test_invalid_epoch_budget(self):
        loader = DataLoader(torch.ones(2, 3), batch_size=2)
        with patch.object(train_utils.torch, 'load') as load:
            with self.assertRaises(ValueError):
                train_utils.train_vae_with_test_set(TinyVAE(), loader, loader, n_epochs=2, warmup_epochs=2)
            load.assert_not_called()

    def test_nonfinite_losses_do_not_load_stale_checkpoint(self):
        loader = DataLoader(torch.ones(2, 3), batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / 'best.pth'
            torch.save(TinyVAE().state_dict(), ckpt)
            previous = ckpt.read_bytes()
            with patch.object(ScheduledLoss, 'values', [float('inf')]), patch.object(train_utils, 'VAELoss', ScheduledLoss):
                with self.assertRaisesRegex(RuntimeError, 'No finite validation loss'):
                    train_utils.train_vae_with_test_set(TinyVAE(), loader, loader,
                        n_epochs=1, warmup_epochs=0, device='cpu', ckpt_path=str(ckpt))
            self.assertEqual(ckpt.read_bytes(), previous)

    def test_zero_warmup_real_training_and_checkpoint_restore(self):
        torch.manual_seed(42)
        model = ConditionalVAE(6, 2, [4], n_classes=0)
        loader = DataLoader(torch.randn(8, 6), batch_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / 'best.pth'
            history = train_utils.train_vae_with_test_set(model, loader, loader,
                n_epochs=2, warmup_epochs=0, device='cpu', ckpt_path=str(ckpt))
            self.assertEqual(len(history['test_loss']), 2)
            for loss, recon, kl in zip(history['test_loss'], history['test_recon'], history['test_kl']):
                self.assertAlmostEqual(loss, recon + kl, places=5)
            for key, value in torch.load(ckpt, weights_only=True).items():
                torch.testing.assert_close(model.state_dict()[key], value)

    def test_legacy_entry_point_uses_shared_loop(self):
        with patch.object(train_utils, 'train_vae_with_test_set', return_value={'shared': True}) as train:
            result = legacy_train(None, None, None, n_epochs=3, warmup_epochs=1, device='cpu')
            self.assertEqual(result, {'shared': True})
            self.assertEqual(train.call_args.kwargs['warmup_epochs'], 1)

if __name__ == '__main__':
    unittest.main()
