import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

temporal_layer = {'LogVarLayer': LogVarLayer}

# Net: 1_2_3
class FBMSTSNet(nn.Module):

    def __init__(self, kernels, fs, temporalLayer = 'LogVarLayer', strideFactor= 5,
                    in_channels:int=22, nbands=12, num_classes=9, radix=8):
        super(FBMSTSNet, self).__init__()
        self.fs = fs
        self.kernels = kernels # type(List); conv_window kernels size
        self.parallel_conv = nn.ModuleList()
        self.strideFactor = strideFactor
        
        # 1D-parallel_conv
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv2d(in_channels=nbands, out_channels=nbands, kernel_size=(1,kernel_size),
                               stride=1, padding=0, bias=False, groups=nbands)
            self.parallel_conv.append(sep_conv)

        self.convblock = nn.Sequential(
            nn.BatchNorm2d(num_features=nbands),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=nbands, out_channels=nbands*radix, kernel_size=(in_channels,1),
                               stride=1, padding=0, bias=False)
            )
        
        self.temporalLayer = temporal_layer[temporalLayer](dim=-1)

        
        self.fc = nn.Sequential(
            nn.Linear(in_features=nbands*radix*strideFactor, out_features=num_classes),
            nn.LogSoftmax(dim=1)
        )
        
    
    def _get_fea_dim(self,x):
        with torch.no_grad():
            features, _ = self.forward(x)
        return features.shape

    def forward(self, x):
        out_sep = []
        # forward paralle 1D-conv blocks
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)
        out = torch.cat(out_sep, dim=-1)
        out = self.convblock(out)
        out = torch.squeeze(out) # N, C', T'

        pad_length = self.strideFactor - (out.shape[-1] % self.strideFactor)
        if pad_length != 0:
            out = F.pad(out, (0, pad_length))

        out = out.reshape([*out.shape[0:2], self.strideFactor, int(out.shape[-1]/self.strideFactor)])
        out = self.temporalLayer(out)
        out = torch.flatten(out, start_dim=1)

        features = out
        return features, self.fc(features)
    
    
import matplotlib.pyplot as plt
import numpy as np
import torch, os

# 假设 weight 是 [out_channels, in_channels, kernel_h, kernel_w]
# 例如 weight = spatial_conv.weight.data.cpu().abs()
# 你的 nbands, in_channels, radix 已知

SAVE_ROOT = "/mnt/data1/tyl/UserID/baseline/figs"
TOP_K = 5

os.makedirs(SAVE_ROOT, exist_ok=True)
model_1 = FBMSTSNet(kernels=[11,21,31,41,51], fs=250, num_classes=20, in_channels=64, nbands=12)
model_2 = FBMSTSNet(kernels=[11,21,31,41,51], fs=250, num_classes=54, in_channels=62, nbands=12)

TASKS = [
    "M3CV_Rest", "M3CV_Transient", "M3CV_Steady",
    "M3CV_P300", "M3CV_Motor", "M3CV_SSVEP_SA",
    "MI", "SSVEP"
]
SEEDS = range(2024, 2029)

MODEL_ROOT = "/mnt/data1/tyl/UserID/ModelSave"

# =========================================================
# Main loop
# =========================================================
for task in TASKS:
    model = model_1 if task.startswith("M3CV") else model_2
    model.eval()
    print(f"\n[Task] {task}")
    task_dir = os.path.join(SAVE_ROOT, task)
    os.makedirs(task_dir, exist_ok=True)

    band_scores_all = []
    chan_scores_all = []
    heatmaps_all = []
    seeds_all = []

    for seed in SEEDS:

        print(f"  Seed {seed}")
        ckpt = f"{MODEL_ROOT}/{task}/CELabelSmooth_FBMSTSNet_{seed}.pth"
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

        # -------------------------
        # Extract weights
        # -------------------------
        spatial_conv = model.convblock[2]
        weight = spatial_conv.weight.data.abs()   # [nbands*radix, nbands, C, 1]

        # -------------------------
        # Heatmap: [nbands, C]
        # -------------------------
        heatmap = weight.sum(dim=0).squeeze()
        heatmap = heatmap / heatmap.max()

        nbands, nchannels = heatmap.shape

        # -------------------------
        # Scores
        # -------------------------
        band_scores = heatmap.mean(dim=1)   # [nbands]
        chan_scores = heatmap.mean(dim=0)   # [C]

        band_scores_all.append(band_scores.cpu().numpy())
        chan_scores_all.append(chan_scores.cpu().numpy())
        heatmaps_all.append(heatmap.cpu().numpy())
        seeds_all.append(seed)

        gc.collect()

    # =====================================================
    # Cross-seed aggregation
    # =====================================================
    band_scores_all = np.stack(band_scores_all, axis=0)   # [S, nbands]
    chan_scores_all = np.stack(chan_scores_all, axis=0)   # [S, C]
    heatmaps_all    = np.stack(heatmaps_all, axis=0)      # [S, nbands, C]

    band_mean = band_scores_all.mean(axis=0)
    band_std  = band_scores_all.std(axis=0)
    chan_mean = chan_scores_all.mean(axis=0)
    chan_std  = chan_scores_all.std(axis=0)

    band_rank = np.argsort(band_mean)[::-1]
    chan_rank = np.argsort(chan_mean)[::-1]

    # =====================================================
    # Representative heatmap (closest to mean)
    # =====================================================
    mean_heatmap = heatmaps_all.mean(axis=0)
    dist = np.linalg.norm(
        heatmaps_all - mean_heatmap[None, :, :],
        axis=(1, 2)
    )
    rep_idx = np.argmin(dist)
    rep_seed = seeds_all[rep_idx]
    rep_heatmap = heatmaps_all[rep_idx]

    # =====================================================
    # Labels
    # =====================================================
    band_labels = [f"{8+2*i}-{10+2*i}Hz" for i in range(nbands)]
    chan_labels = [f"Ch{i+1}" for i in range(nchannels)]

    # =====================================================
    # Save ranking (mean ± std)
    # =====================================================
    rank_path = os.path.join(task_dir, "mean_std_ranking.txt")
    with open(rank_path, "w") as f:

        f.write("Band-wise Top-K (mean ± std)\n")
        for i in range(TOP_K):
            b = band_rank[i]
            f.write(
                f"{i+1}. {band_labels[b]} : "
                f"{band_mean[b]:.4f} ± {band_std[b]:.4f}\n"
            )

        f.write("\nChannel-wise Top-K (mean ± std)\n")
        for i in range(TOP_K):
            c = chan_rank[i]
            f.write(
                f"{i+1}. {chan_labels[c]} : "
                f"{chan_mean[c]:.4f} ± {chan_std[c]:.4f}\n"
            )

        f.write(f"\nRepresentative seed: {rep_seed}\n")

    # =====================================================
    # Save representative heatmap
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    im = ax.imshow(rep_heatmap, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="Normalized Weight")

    ax.set_yticks(np.arange(nbands))
    ax.set_yticklabels(band_labels, fontsize=9)
    ax.set_xticks(np.arange(nchannels))
    ax.set_xticklabels(chan_labels, rotation=90, fontsize=7)

    ax.set_xlabel("EEG Channel")
    ax.set_ylabel("Subband")
    ax.set_title(f"{task} | Representative Seed = {rep_seed}")

    fig.tight_layout()
    plt.savefig(
        os.path.join(task_dir, "representative_heatmap.png"),
        dpi=400,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"  ✔ Saved summary for {task}")

print("\n[Done] All tasks processed.")
