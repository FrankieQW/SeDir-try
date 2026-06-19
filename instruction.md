# SeDiR Environment and Training Instructions

This project is a SeDiR reproduction scaffold derived from MC3D-AD. It keeps the MC3D-AD dataset, PointMAE backbone, training loop, and evaluation pipeline, then replaces the reconstruction model with SeDiR-style CFGT, C3L, and GGD modules.

## 1. Create Conda Environment

```powershell
conda create -n SeDir python=3.8 -y
conda activate SeDir
```

Install PyTorch for CUDA 11.7:

```powershell
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Optional test helper:

```powershell
pip install pytest
```

## 2. Pretrained PointMAE Checkpoint

Download `modelnet_8k.pth` from the official Point-MAE release and place it here:

```text
pretrain_ckp/modelnet_8k.pth
```

The default configs already point to:

```text
../../pretrain_ckp/modelnet_8k.pth
```

## 3. Dataset Paths

Edit these fields before training:

```yaml
dataset:
  data_dir: /absolute/path/to/your/dataset

net:
  - name: backbone
    kwargs:
      data_dir: /absolute/path/to/your/dataset
```

For Real3D-AD, use the downsampled `real3D_down` directory created by:

```powershell
python downsample_pcd.py --real3d_path <Path/to/your/Real3D-AD-PCD>
```

For Anomaly-ShapeNet, arrange the dataset in the same layout expected by MC3D-AD.

## 4. SeDiR Configuration

The reproduction config uses the paper-style settings:

- PointMAE pretrained on ModelNet40 8k
- frozen backbone
- batch size 1
- 1024 groups
- hidden dimension 256
- 4 encoder layers and 4 decoder layers
- 8 attention heads
- C3L buffer size 64
- `lambda_scl=0.001`
- `lambda_cls=0.001`
- `lambda_cos=0.01`

The main model is registered as:

```yaml
type: models.reconstructions.SeDiR
```

The total objective is:

```yaml
type: SeDiRLoss
```

## 5. Train

Real3D-AD:

```powershell
cd experiments\real3d
conda activate SeDir
python ..\..\tools\train_val.py --config config.yaml
```

Anomaly-ShapeNet:

```powershell
cd experiments\Anomaly_ShapeNet
conda activate SeDir
python ..\..\tools\train_val.py --config config.yaml
```

If you use the original shell scripts, check that the GPU id and Python environment match your machine.

## 6. Evaluate

Set `saver.load_path` in the relevant `config.yaml` to the checkpoint you want to evaluate, then run the original evaluation command from the experiment folder.

## 7. 6GB GPU Notes

The default SeDiR configs use 1024 groups because the paper reports 1024 groups. This is much more practical on a 6GB RTX4050 Laptop than the MC3D-AD Real3D config with 4096 groups.

If CUDA memory is insufficient:

1. Keep batch size at 1.
2. Keep the PointMAE backbone frozen.
3. Use `num_group: 512` only for debugging.
4. Reduce `num_encoder_layers` and `num_decoder_layers` to 2 for smoke tests.
5. Re-enable paper settings for final runs.

## 8. Smoke Test

Without real data, you can verify the SeDiR module with:

```powershell
conda activate SeDir
python -c "import torch; from models.reconstructions.sedir import SeDiR; m=SeDiR(inplanes=384,feature_size=16,hidden_dim=64,nhead=4,num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=128,dropout=0.0,cls_num=4,feature_jitter=None,neighbor_mask=None,initializer={'method':'xavier_uniform'}); x={'xyz_features':torch.randn(2,384,16),'center':torch.randn(2,16,3),'cls_label':torch.tensor([0,1]),'filename':['a','b']}; y=m(x); loss=y['loss_rec']+y['loss_scl']+y['loss_cls']+y['loss_cos']; loss.backward(); print(y['feature_rec'].shape, y['pred'].shape)"
```

Expected output:

```text
torch.Size([2, 384, 16]) torch.Size([2, 1, 16])
```

## 9. Reproduction Caveat

This is a faithful engineering reproduction based on the SeDiR paper text and the MC3D-AD codebase. Exact AUROC matching is not guaranteed without the official SeDiR implementation, random seeds, preprocessing details, and released checkpoints.
