import argparse
import os
import sys
import time
import gc
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_curve
from sklearn.preprocessing import OneHotEncoder

sys.path.append("/mnt/data1/tyl/UserID/")
warnings.filterwarnings("ignore")

from baseline.dataloader import *
from dataset.instance import set_seed
from utils.utlis import load_baseline_model, load_data, set_args
from utils.mylogging import Logger


# =========================
# EER (kept close to your version)
# NOTE: uses predicted labels as "scores" (not a true score-based EER).
# =========================
def calculate_eer(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    classes = np.unique(y_true)
    n_classes = len(classes)
    if n_classes <= 1:
        return 0.0, [0.0]

    encoder = OneHotEncoder(sparse_output=False)
    y_true_bin = encoder.fit_transform(y_true.reshape(-1, 1))
    y_pred_bin = encoder.transform(y_pred.reshape(-1, 1))

    class_eer = []
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        fnr = 1 - tpr
        idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = float(np.mean([fpr[idx], fnr[idx]]))
        class_eer.append(eer)

    avg_eer = float(np.mean(class_eer)) if len(class_eer) > 0 else 0.0
    return avg_eer, class_eer


# =========================
# Noise Injection
# x: torch.Tensor, shape (B,1,C,T)
# noisy_type: ["none", "temporal", "spatial", "zero"]
# =========================
def add_noisy(
    x: torch.Tensor,
    noisy_type: str = "none",
    noise_std_scale: float = 1.0,
    temporal_ratio: float = 0.5,
    temporal_mode: str = "middle",
    generator: torch.Generator = None,
):
    if noisy_type is None or noisy_type.lower() == "none":
        return x

    assert x.dim() == 4 and x.size(1) == 1, f"Expected (B,1,C,T), got {tuple(x.shape)}"
    B, _, C, T = x.shape
    device = x.device
    dtype = x.dtype

    noisy_type = noisy_type.lower()
    x_noisy = x.clone()

    mean = x_noisy[:, 0, :, :].mean(dim=-1)  # (B,C)
    std = x_noisy[:, 0, :, :].std(dim=-1).clamp_min(1e-6)  # (B,C)
    amp = (mean.abs() + std) * noise_std_scale  # (B,C)

    if noisy_type == "temporal":
        seg_len = max(1, int(T * temporal_ratio))
        if temporal_mode == "middle":
            start = (T - seg_len) // 2
        else:
            # random start
            high = max(1, T - seg_len + 1)
            start = int(torch.randint(0, high, (1,), device=device, generator=generator).item())
        end = start + seg_len

        amp_bc = amp.view(B, 1, C, 1).to(dtype=dtype)
        noise = torch.randn((B, 1, C, seg_len), device=device, dtype=dtype, generator=generator) * amp_bc
        x_noisy[:, :, :, start:end] = x_noisy[:, :, :, start:end] + noise
        return x_noisy

    elif noisy_type == "spatial":
        ch_idx = torch.randint(0, C, (B,), device=device, generator=generator)
        amp_sel = amp[torch.arange(B, device=device), ch_idx].to(dtype=dtype)  # (B,)
        noise_bt = torch.randn((B, T), device=device, dtype=dtype, generator=generator) * amp_sel.unsqueeze(-1)

        for b in range(B):
            c = int(ch_idx[b].item())
            x_noisy[b, 0, c, :] = x_noisy[b, 0, c, :] + noise_bt[b]
        return x_noisy

    elif noisy_type == "zero":
        ch_idx = torch.randint(0, C, (B,), device=device, generator=generator)
        for b in range(B):
            c = int(ch_idx[b].item())
            x_noisy[b, 0, c, :] = 0.0
        return x_noisy

    else:
        raise ValueError(f"Unknown noisy_type: {noisy_type}")


def _make_noise_generator(device: torch.device, seed: int, noisy_type: str):
    """
    Random but reproducible generator for test-time noise:
    same (seed, noisy_type) => same noise stream (given same data order).
    """
    if noisy_type is None or noisy_type.lower() == "none":
        return None
    type2id = {"temporal": 11, "spatial": 22, "zero": 33}
    tid = type2id.get(noisy_type.lower(), 99)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) * 1000 + tid)
    return gen


def evaluate(model, dataloader, args, noisy_type="none"):
    model.eval()
    correct, loss_sum = 0.0, 0.0
    total = len(dataloader.dataset)

    device = torch.device("cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu")
    clf_loss_func = nn.CrossEntropyLoss().to(device)

    y_pred, y_true = [], []

    gen = _make_noise_generator(device=device, seed=args.seed, noisy_type=noisy_type)

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        if gen is not None:
            x = add_noisy(
                x,
                noisy_type=noisy_type,
                noise_std_scale=args.noise_std_scale,
                temporal_ratio=args.temporal_ratio,
                temporal_mode=args.temporal_mode,
                generator=gen,
            )

        with torch.no_grad():
            _, logits = model(x)
            pred_y = torch.max(logits, 1)[1]
            loss = clf_loss_func(logits, y.long())

        correct += float((pred_y == y).sum())
        loss_sum += float(loss.item())

        y_pred.extend(pred_y.detach().long().cpu().tolist())
        y_true.extend(y.detach().long().cpu().tolist())

    avg_loss = loss_sum / max(1, len(dataloader))
    acc = correct / max(1, total)
    f1 = f1_score(np.array(y_true), np.array(y_pred), average="weighted")
    eer, _ = calculate_eer(np.array(y_true), np.array(y_pred))
    return avg_loss, acc, f1, eer


def train(trainloader, valloader, savepath, args):
    print("-" * 20 + "开始训练!" + "-" * 20)
    device, model, optimizer = load_baseline_model(args)
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)

    clf_loss_func = nn.CrossEntropyLoss().to(device)

    best_epoch = 0
    best_acc = 0.0

    for epoch in tqdm(range(args.epoch), desc="Training:"):
        model.train()
        train_correct = 0.0
        one_epoch_loss = 0.0

        for _, (b_x, b_y) in enumerate(trainloader):
            optimizer.zero_grad()
            b_x, b_y = b_x.to(device), b_y.to(device)

            _, output = model(b_x)
            loss = clf_loss_func(output, b_y.long())
            one_epoch_loss += float(loss.item())

            loss.backward()
            optimizer.step()

            pred_cls = torch.argmax(output, dim=1)
            train_correct += float((pred_cls == b_y).sum())

        train_acc = train_correct / max(1, len(trainloader.dataset))

        with torch.no_grad():
            val_loss, val_acc, val_f1, val_eer = evaluate(model, valloader, args, noisy_type="none")

        # early stop
        if (epoch - best_epoch) > args.earlystop:
            break

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            # 模型保存：带 noise_type 仅用于区分“本次实验配置”，不代表训练加噪
            torch.save(
                model.state_dict(),
                os.path.join(savepath, f"Noisy_{args.noise_type}_{args.model}_session{args.session_num}_{args.seed}.pth"),
            )

        if (epoch + 1) % 10 == 0:
            one_epoch_loss_avg = one_epoch_loss / max(1, len(trainloader))
            print(
                f"Epoch:{epoch+1}\tTrain_acc:{train_acc:.4f}\tVal_acc:{val_acc:.4f}\t"
                f"Train_loss:{one_epoch_loss_avg:.6f}\tVal_loss:{val_loss:.6f}"
            )

    print("-" * 20 + "训练完成!" + "-" * 20)
    print(f"总训练轮数-{epoch+1}, 早停轮数-{best_epoch+1}")
    print(f"验证集最佳准确率为{best_acc*100:.2f}%")
    return model


def main():
    parser = argparse.ArgumentParser(description="Model Train Hyperparameter")
    parser.add_argument("--root", type=str, default="/mnt/data1/tyl/UserID/dataset/mydata")
    parser.add_argument("--setsplit", type=str, default=None)
    parser.add_argument("--gpuid", type=int, default=9)
    parser.add_argument("--nclass", type=int, default=9)
    parser.add_argument("--channel", type=int, default=22)
    parser.add_argument("--timepoint", type=int, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--initlr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--earlystop", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--model", type=str, default="ResEEGNet")
    parser.add_argument("--strideFactor", type=int, default=4, help="尺度因子数")
    parser.add_argument("--session_num", type=int, default=1, help="跨任务中使用第几个session数据")
    parser.add_argument("--cross_task", type=str, nargs="+", default=None, help="跨任务")
    parser.add_argument("--tl", type=str, default=None, help="迁移学习方法")

    # ===== noise args =====
    # noise_type: none / temporal / spatial / zero / all
    parser.add_argument("--noise_type", type=str, default="all",
                        help="test-time noise type: none|temporal|spatial|zero|all")
    parser.add_argument("--noise_std_scale", type=float, default=1.0, help="Gaussian noise amplitude scale")
    parser.add_argument("--temporal_ratio", type=float, default=0.5, help="temporal noise segment ratio of T")
    parser.add_argument("--temporal_mode", type=str, default="middle", choices=["middle", "random"],
                        help="temporal segment placement")

    args = parser.parse_args()
    set_args(args)

    # ===== cross_task name =====
    cross_task = "unknown"
    if args.cross_task is not None and args.cross_task[0].startswith("M3CV"):
        cross_task = args.cross_task[0][5:] + "_" + args.cross_task[1][5:]
    elif args.cross_task is not None and args.cross_task[0] in ["MI", "ERP", "SSVEP", "CatERP"]:
        cross_task = args.cross_task[0] + "_" + args.cross_task[1]

    # ===== logging =====
    log_name = f"{cross_task}_{args.model}_Noisy_{args.noise_type}_session{args.session_num}_base.log"
    sys.stdout = Logger(os.path.join("/mnt/data1/tyl/UserID/logdir", log_name))

    # ===== decide eval noise list =====
    nt = (args.noise_type or "none").lower()
    if nt == "all":
        eval_noise_list = ["temporal", "spatial", "zero"]
    elif nt in ["none"]:
        eval_noise_list = []
    elif nt in ["temporal", "spatial", "zero"]:
        eval_noise_list = [nt]
    else:
        raise ValueError(f"--noise_type must be one of none|temporal|spatial|zero|all, got {args.noise_type}")

    results_all = []  # per-seed dict

    for seed in range(2024, 2029):
        args.seed = seed
        start_time = time.time()

        print("=" * 30)
        print(f"dataset: {cross_task}")
        print(f"model  : {args.model}")
        print(f"seed   : {args.seed}")
        print(f"gpu    : {args.gpuid}")
        print(f"noise  : {args.noise_type}")

        set_seed(args.seed)
        trainloader, valloader, testloader = load_data(args)

        print("=====================data are prepared===============")
        print(f"累计用时{time.time()-start_time:.4f}s!")

        model_path = os.path.join("/mnt/data1/tyl/UserID/ModelSave", f"{cross_task}")
        os.makedirs(model_path, exist_ok=True)

        model = train(trainloader, valloader, model_path, args)
        print("=====================model are trained===============")
        print(f"累计用时{time.time()-start_time:.4f}s!")

        # ===== test: clean + selected noise(s) =====
        clean_loss, clean_acc, clean_f1, clean_eer = evaluate(model, testloader, args, noisy_type="none")
        print(f"[CLEAN] Acc:{clean_acc*100:.2f}%  F1:{clean_f1*100:.2f}%  EER:{clean_eer*100:.2f}%")

        row = {
            "seed": seed,
            "clean_acc": clean_acc, "clean_f1": clean_f1, "clean_eer": clean_eer,
        }

        for ntype in eval_noise_list:
            _, nacc, nf1, neer = evaluate(model, testloader, args, noisy_type=ntype)
            print(f"[{ntype.upper()}] Acc:{nacc*100:.2f}%  F1:{nf1*100:.2f}%  EER:{neer*100:.2f}%")
            row[f"{ntype}_acc"] = nacc
            row[f"{ntype}_f1"] = nf1
            row[f"{ntype}_eer"] = neer

        results_all.append(row)

        print("=====================test are done===================")
        print(f"训练集:验证集:测试集={len(trainloader.dataset)}:{len(valloader.dataset)}:{len(testloader.dataset)}")
        gc.collect()

    # ===== save CSV: per seed + Avg/Std =====
    df = pd.DataFrame(results_all).set_index("seed")

    df_avg = df.mean(axis=0).to_frame().T
    df_std = df.std(axis=0).to_frame().T
    df_avg.index = ["Avg"]
    df_std.index = ["Std"]
    df_final = pd.concat([df, df_avg, df_std], axis=0).round(4)

    csv_path = os.path.join("/mnt/data1/tyl/UserID/csv", f"{cross_task}")
    os.makedirs(csv_path, exist_ok=True)

    out_name = f"Noisy_{args.noise_type}_{args.model}_session{args.session_num}.csv"
    out_file = os.path.join(csv_path, out_name)
    df_final.to_csv(out_file)

    print("-" * 50)
    print("Saved:", out_file)


if __name__ == "__main__":
    main()
