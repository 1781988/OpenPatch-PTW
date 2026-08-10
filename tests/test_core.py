import torch

from openpatch_ptw.attacks import copy_move, cross_image_patch_transfer, residual_transfer
from openpatch_ptw.heads import LocalCodeHead, OpenSetStatusHead, consistency_map
from openpatch_ptw.position_code import PositionBoundSpatialInjection, PositionCodeField


def test_position_code_changes_with_coordinates_and_message():
    torch.manual_seed(0)
    gen = PositionCodeField(bit_dim=8, code_dim=4, hidden_dim=16, fourier_bands=2)
    bits0 = torch.zeros(2, 8)
    bits1 = torch.ones(2, 8)
    c0 = gen(bits0, (16, 16))
    c1 = gen(bits1, (16, 16))
    assert c0.shape == (2, 4, 16, 16)
    assert not torch.allclose(c0[:, :, 0, 0], c0[:, :, -1, -1])
    assert not torch.allclose(c0, c1)


def test_position_injection_small_residual_at_initialization():
    torch.manual_seed(0)
    layer = PositionBoundSpatialInjection(
        wm_latent_dim=4 * 8 * 8,
        z_channels=16,
        bit_dim=8,
        code_dim=4,
        hidden_dim=16,
    )
    z = torch.randn(2, 16, 32, 32)
    wm = torch.randn(2, 4 * 8 * 8)
    bits = torch.randint(0, 2, (2, 8)).float()
    out, code = layer(z, wm, bits)
    assert out.shape == z.shape
    assert code.shape == (2, 4, 32, 32)
    # final residual conv is zero-initialized
    assert torch.allclose(out, z, atol=1e-6)


def test_consistency_and_status_shapes():
    feat = torch.randn(2, 1, 32, 32)
    code_head = LocalCodeHead(code_dim=4)
    pred_code = code_head(feat)
    residual = consistency_map(pred_code, torch.zeros_like(pred_code))
    status = OpenSetStatusHead()(feat, torch.rand(2, 64), residual)
    assert residual.shape == (2, 1, 32, 32)
    assert status.shape == (2, 3)


def test_forgery_attack_shapes():
    x = torch.zeros(2, 3, 32, 32)
    donor = torch.ones_like(x) * 0.1
    wm = donor + 0.01
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, 8:16, 8:16] = 1
    assert residual_transfer(x, donor, wm).image.shape == x.shape
    assert cross_image_patch_transfer(x, donor, mask).mask.shape == mask.shape
    assert copy_move(donor, mask).image.shape == x.shape
