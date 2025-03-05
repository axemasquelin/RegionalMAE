import unittest
import torch
import os, sys
from lib.networks.MaskedViTv2 import MAE

import torch.nn.functional as F

class TestMaskedViTv2(unittest.TestCase):
    def setUp(self):
        self.model = MAE(img_size=64, patch_size=8, chn_in=1, heads=12, embed_dim=768, dec_embedim=3072, mlp_dim=2048, encoder_layers=6, decoder_layers=4, mask_ratio=0.75)
        self.img = torch.randn(2, 1, 64, 64)  # Batch of 2 images, 1 channel, 64x64 size
        self.labels = torch.randint(0, 2, (2, 2)).float()  # Binary labels for classification

    def test_patchify_unpatchify(self):
        patches = self.model.patchify(self.img)
        reconstructed_img = self.model.unpatchify(patches)
        self.assertEqual(self.img.shape, reconstructed_img.shape)

    def test_random_masking(self):
        patches = self.model.patchify(self.img)
        masked_patches, mask = self.model.random_masking(patches, self.model.mask_ratio)
        self.assertEqual(masked_patches.shape, patches.shape)
        self.assertEqual(mask.shape, (patches.shape[0], patches.shape[1]))

    def test_reconstruction_loss(self):
        patches = self.model.patchify(self.img)
        pred_patches = torch.randn_like(patches)
        mask = torch.ones(patches.shape[0], patches.shape[1])
        loss = self.model.reconstruction_loss(pred_patches, patches, mask)
        self.assertIsInstance(loss, torch.Tensor)

    def test_forward_reconstruction(self):
        loss = self.model(self.img, task='reconstruction')
        self.assertIsInstance(loss, torch.Tensor)

    def test_forward_classification(self):
        output, loss = self.model(self.img, labels=self.labels, task='dx')
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(loss, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
