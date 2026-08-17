import torch

from openpatch_ptw.attacks import copy_move, cross_image_patch_transfer, residual_transfer
from openpatch_ptw.heads import LocalCodeHead, OpenSetStatusHead, consistency_map
from openpatch_ptw.losses import mask_loss
from openpatch_ptw.masks import generate_multiscale_mask
from openpatch_ptw.position_code import PositionBoundSpatialInjection, PositionCodeField


def test_position_code_changes_with_coordinates_and_message():
    generator = PositionCodeField(bit_dim=8, code_dim=4, fourier_bands=2)
    zeros = torch.zeros(2, 8)
    ones = torch.ones(2, 8)
    code0 = generator(zeros, (16, 16))
    code1 = generator(ones, (16, 16))
    assert code0.shape == (2, 4, 16, 16)
    assert not torch.allclose(code0[:, :, 0, 0], code0[:, :, -1, -1])
    assert not torch.allclose(code0, code1)


def test_position_branch_is_zero_at_initialization():
    layer = PositionBoundSpatialInjection(
        wm_latent_dim=4 * 8 * 8,
        z_channels=16,
        bit_dim=8,
        code_dim=4,
    )
    layer.eval()
    z = torch.randn(2, 16, 32, 32)
    wm = torch.randn(2, 4 * 8 * 8)
    bits = torch.randint(0, 2, (2, 8)).float()
    output, code = layer(z, wm, bits)
    # Base and position branches are neutral before official warm-start.
    assert torch.allclose(output, z, atol=1e-6)
    assert code.shape == (2, 4, 32, 32)


def test_consistency_status_and_mask_loss_are_finite():
    feature = torch.randn(2, 1, 32, 32)
    code_head = LocalCodeHead(code_dim=4)
    predicted = code_head(feature)
    residual = consistency_map(predicted, torch.zeros_like(predicted))
    status = OpenSetStatusHead()(feature, torch.rand(2, 64), residual)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:16, 8:16] = 1
    loss = mask_loss(torch.randn_like(target), target, edge_weight=2.0)
    assert residual.shape == (2, 1, 32, 32)
    assert status.shape == (2, 3)
    assert torch.isfinite(loss)


def test_forgery_attack_shapes():
    target = torch.zeros(2, 3, 32, 32)
    donor = torch.ones_like(target) * 0.1
    watermarked = donor + 0.01
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, 8:16, 8:16] = 1
    assert residual_transfer(target, donor, watermarked).image.shape == target.shape
    assert cross_image_patch_transfer(target, donor, mask).mask.shape == mask.shape
    assert copy_move(donor, mask).image.shape == target.shape


def test_multiscale_mask_is_deterministic_and_close_to_bin():
    mask1 = generate_multiscale_mask(128, [[0.05, 0.10, 1.0]], seed=123)
    mask2 = generate_multiscale_mask(128, [[0.05, 0.10, 1.0]], seed=123)
    assert torch.equal(mask1, mask2)
    assert 0.02 <= float(mask1.mean()) <= 0.14
