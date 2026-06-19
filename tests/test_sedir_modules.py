import torch

from models.reconstructions.sedir import SeDiR


def make_batch(batch_size=2, channels=384, groups=16, classes=4):
    labels = torch.arange(batch_size) % classes
    return {
        "xyz_features": torch.randn(batch_size, channels, groups),
        "center": torch.randn(batch_size, groups, 3),
        "clsname": [str(int(label)) for label in labels],
        "category": labels,
        "filename": [f"sample_{idx}.pcd" for idx in range(batch_size)],
    }


def test_sedir_forward_returns_reconstruction_scores_and_losses():
    model = SeDiR(
        inplanes=384,
        feature_size=16,
        hidden_dim=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        cls_num=4,
        feature_jitter=None,
        neighbor_mask=None,
        initializer={"method": "xavier_uniform"},
        c3l_buffer_size=8,
        contrast_temperature=0.2,
    )

    output = model(make_batch())

    assert output["feature_rec"].shape == (2, 384, 16)
    assert output["feature_align"].shape == (2, 384, 16)
    assert output["pred"].shape == (2, 1, 16)
    assert output["cls_pred"].shape == (2, 4)
    assert output["global_token"].shape[0] == 2
    assert set(["loss_scl", "loss_cls", "loss_cos", "loss_rec"]).issubset(output)


def test_sedir_losses_backpropagate():
    model = SeDiR(
        inplanes=384,
        feature_size=16,
        hidden_dim=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        cls_num=4,
        feature_jitter=None,
        neighbor_mask=None,
        initializer={"method": "xavier_uniform"},
        c3l_buffer_size=8,
        contrast_temperature=0.2,
    )

    output = model(make_batch())
    loss = output["loss_rec"] + output["loss_scl"] + output["loss_cls"] + output["loss_cos"]
    loss.backward()

    grad_norm = model.input_proj.weight.grad.abs().sum().item()
    assert grad_norm > 0
