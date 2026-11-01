# Дополнительная ячейка для soi_research_lab.ipynb.
# Выполните основные определения старого ноутбука, затем эту ячейку.
# Обучение запускается отдельной кнопкой; старые результаты не перезаписываются.

_jw_required = ["Denoiser", "evaluate", "covariance", "gaussian_truth",
                "atomic_json", "atomic_csv", "sync_device", "training_plot",
                "VERSION", "OUTPUT_ROOT"]
_jw_missing = [name for name in _jw_required if name not in globals()]
if _jw_missing:
    raise RuntimeError("Сначала выполните определения исходного ноутбука. Нет: "
                       + ", ".join(_jw_missing))
if VERSION != "soi-ablation-1.0":
    raise RuntimeError("Эта ячейка проверена с soi-ablation-1.0; версия отличается.")

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import ipywidgets as widgets
from IPython.display import display, FileLink
from tqdm.auto import tqdm

JW_VERSION = "joint-weight-1.0"
JW_CASES = ["balanced"]
JW_SEEDS = [5, 6, 7, 8, 9]
JW_CONFIG = dict(
    width=64, sigma_min=.005, sigma_max=100., grid_size=48, eval_batch=256,
    steps=8000, lr=.002, batch_size=64, log_every=250, validate_every=500,
    n_train=15000, n_val=256, n_test=8192,
)
# n_test увеличен только для оценки готовых сетей: обучение от него не зависит.
# Веса заранее фиксированы: 1 (контроль) и 2 (гипотеза). Checkpoint — шаг 8000.
# Не выбирайте лучший checkpoint/вес по финальному тесту.


def jw_run_one(cfg, case, seed, joint_weight, root, device, live=True):
    """Та же модель и RNG, что у shared; изменена только формула loss."""
    cov = covariance(case)
    n = len(cov)
    job = dict(version=JW_VERSION, base_version=VERSION, case=case, seed=int(seed),
               method="shared", joint_weight=float(joint_weight), config=cfg,
               torch_version=str(torch.__version__), device=str(device))
    token = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:12]
    folder = Path(root) / f"{case}_s{seed}_w{joint_weight:g}_{token}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        print("Готово ранее:", folder.name)
        return json.loads((folder / "result.json").read_text())
    atomic_json(folder / "config.json", job)

    torch.manual_seed(1000 + seed)
    model = Denoiser(n, cfg["width"], False, cfg["sigma_min"], cfg["sigma_max"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    x_train = np.random.default_rng(10000 + seed).multivariate_normal(
        np.zeros(n), cov, cfg["n_train"]).astype(np.float32)
    x_val = np.random.default_rng(20000 + seed).multivariate_normal(
        np.zeros(n), cov, cfg["n_val"])
    x_test = np.random.default_rng(30000 + seed).multivariate_normal(
        np.zeros(n), cov, cfg["n_test"])
    rng = np.random.default_rng(40000 + seed)
    checkpoint = folder / "checkpoint.pt"
    start, history, elapsed = 0, [], 0.
    if checkpoint.exists():
        # Загружается только собственный checkpoint из папки этого запуска.
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["rng_state"]
        start, history, elapsed = saved["step"], saved["history"], saved["train_seconds"]
        print("Продолжение с шага", start)

    progress = tqdm(range(start, cfg["steps"]), total=cfg["steps"], initial=start,
                    desc=f"{case}, seed={seed}, joint weight={joint_weight:g}")
    totals, count, handle = np.zeros(4), 0, None
    sync_device(device)
    clock = time.perf_counter()
    for step0 in progress:
        model.train()
        ids = rng.integers(cfg["n_train"], size=cfg["batch_size"])
        noise = rng.normal(size=(cfg["batch_size"], n)).astype(np.float32)
        u = rng.uniform(np.log(cfg["sigma_min"]), np.log(cfg["sigma_max"]),
                        size=(cfg["batch_size"], 1))
        x = torch.as_tensor(x_train[ids], device=device)
        z = torch.as_tensor(noise, device=device)
        sigma = torch.as_tensor(np.exp(u).astype(np.float32), device=device)
        j, m, c = model(x, z, sigma)
        rj, rm, rc = [(a-z).square().mean() for a in (j, m, c)]
        loss = (joint_weight * rj + rm + rc) / (joint_weight + 2)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Нечисловой loss: {folder}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        totals += torch.stack([loss, rj, rm, rc]).detach().cpu().numpy()
        count += 1
        step = step0 + 1
        if step % cfg["log_every"] == 0 or step == cfg["steps"]:
            sync_device(device)
            elapsed += time.perf_counter() - clock
            row = dict(step=step, train_seconds=elapsed,
                       **dict(zip(["loss", "loss_joint", "loss_marginal",
                                   "loss_conditional"], (totals/count).tolist())))
            if step % cfg["validate_every"] == 0 or step == cfg["steps"]:
                val, _, _ = evaluate(model, x_val, cov, cfg, grid_size=16, seed=50000+seed)
                for name in ["direct", "risk", "error_joint", "error_marginal",
                             "error_conditional", "predicted_risk_bias"]:
                    row["val_"+name] = val[name]
            history.append(row)
            totals, count = np.zeros(4), 0
            tmp = checkpoint.with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            rng_state=rng.bit_generator.state, step=step, history=history,
                            train_seconds=elapsed), tmp)
            os.replace(tmp, checkpoint)
            atomic_csv(folder / "training.csv", pd.DataFrame(history))
            progress.set_postfix(loss=round(row["loss"], 4))
            if "val_direct" in row:
                fig = training_plot(history, gaussian_truth(cov)["omega"], folder / "training.png")
                if live:
                    if handle is None:
                        handle = display(fig, display_id=True)
                    else:
                        handle.update(fig)
                plt.close(fig)
            sync_device(device)
            clock = time.perf_counter()

    summary, curves, observations = evaluate(model, x_test, cov, cfg, seed=60000+seed)
    summary.update(case=case, seed=int(seed), method="shared", joint_weight=float(joint_weight),
                   n_train=cfg["n_train"], n_test=cfg["n_test"], steps=cfg["steps"],
                   parameters=sum(p.numel() for p in model.parameters()),
                   train_seconds=elapsed, folder=str(folder))
    for estimator in ("direct", "risk"):
        error = summary[estimator] - summary["truth_omega"]
        summary[estimator+"_signed_error"] = error
        summary[estimator+"_abs_error"] = abs(error)
        summary[estimator+"_sq_error"] = error**2
    summary["bias_bound"] = (2 * summary["error_joint"] + summary["error_marginal"]
                              + summary["error_conditional"])
    # Оракул используется только при оценке, не в loss и не для выбора checkpoint.
    summary["predicted_risk_bias_abs"] = abs(summary["predicted_risk_bias"])
    atomic_csv(folder / "integrands.csv", curves)
    np.savez_compressed(folder / "observations.npz", **observations)
    atomic_json(folder / "result.json", summary)
    return summary


def jw_tables(results, root):
    """Агрегаты и парные разности; положительный gain означает преимущество w=2."""
    df = pd.DataFrame(results)
    summary = df.groupby(["case", "joint_weight"], as_index=False).agg(
        seeds=("seed", "count"), risk_mae=("risk_abs_error", "mean"),
        risk_mse=("risk_sq_error", "mean"), risk_bias=("risk_signed_error", "mean"),
        direct_mae=("direct_abs_error", "mean"), direct_mse=("direct_sq_error", "mean"),
        direct_bias=("direct_signed_error", "mean"),
        error_joint=("error_joint", "mean"), error_marginal=("error_marginal", "mean"),
        error_conditional=("error_conditional", "mean"),
        predicted_risk_bias=("predicted_risk_bias", "mean"),
        predicted_risk_bias_abs=("predicted_risk_bias_abs", "mean"),
        bias_bound=("bias_bound", "mean"), train_seconds=("train_seconds", "mean"))
    for name in ("direct", "risk"):
        summary[name+"_rmse"] = np.sqrt(summary.pop(name+"_mse"))
    pair = df.loc[df.joint_weight == 1].merge(
        df.loc[df.joint_weight == 2], on=["case", "seed"],
        suffixes=("_w1", "_w2"), validate="one_to_one")
    paired = pair[["case", "seed"]].copy()
    for metric in ["risk_abs_error", "direct_abs_error", "error_joint", "error_marginal",
                   "error_conditional", "predicted_risk_bias", "predicted_risk_bias_abs", "bias_bound"]:
        for suffix in ("w1", "w2"):
            paired[metric+"_"+suffix] = pair[metric+"_"+suffix]
    paired["risk_gain"] = pair.risk_abs_error_w1 - pair.risk_abs_error_w2
    paired["direct_gain"] = pair.direct_abs_error_w1 - pair.direct_abs_error_w2
    paired["joint_error_reduction"] = pair.error_joint_w1 - pair.error_joint_w2
    # SE парной разности оценок Ω на одном тесте и одних шумах.
    # Это НЕ SE разности абсолютных ошибок и НЕ разброс обучения по seed.
    for estimator in ("risk", "direct"):
        deltas, errors = [], []
        for p in pair.itertuples():
            with np.load(Path(p.folder_w1)/"observations.npz") as a, \
                    np.load(Path(p.folder_w2)/"observations.npz") as b:
                delta = b[estimator] - a[estimator]
                deltas.append(float(delta.mean()))
                errors.append(float(delta.std(ddof=1)/np.sqrt(len(delta))))
        paired[estimator+"_delta_w2_minus_w1"] = deltas
        paired[estimator+"_paired_eval_se"] = errors
    atomic_csv(root / "results.csv", df)
    atomic_csv(root / "summary.csv", summary)
    atomic_csv(root / "paired.csv", paired)
    return df, summary, paired


def jw_pair_plot(paired, root, live=True):
    if paired.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in zip(axes, ["risk_gain", "direct_gain"], ["risk", "direct"]):
        for case, group in paired.groupby("case"):
            ax.scatter(group.seed, group[metric], label=case)
        ax.axhline(0, color="black", linewidth=1)
        ax.set(xlabel="Seed", ylabel="Absolute-error reduction, nats",
               title=f"{label}: positive = weight 2 is better")
        ax.set_xticks(sorted(paired.seed.unique()))
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / "paired_comparison.png", dpi=150)
    if live:
        display(fig)
    plt.close(fig)


def jw_run_suite(cfg, cases, seeds, root, device, live=True):
    root = Path(root)
    protocol = dict(version=JW_VERSION, config=cfg, cases=list(cases), seeds=list(seeds),
                    weights=[1, 2], device=str(device), torch=str(torch.__version__))
    token = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    suite_root = root / token
    suite_root.mkdir(parents=True, exist_ok=True)
    atomic_json(suite_root / "protocol.json", protocol)
    results = []
    for case in cases:
        for seed in seeds:
            # Чередуем порядок, чтобы нагрев устройства не всегда приходился на один вес.
            for weight in ([1, 2] if seed % 2 == 0 else [2, 1]):
                try:
                    results.append(jw_run_one(cfg, case, seed, weight, suite_root, device, live))
                except Exception as error:
                    with (suite_root / "failures.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(dict(case=case, seed=seed, weight=weight,
                                                error=repr(error)), ensure_ascii=False)+"\n")
                    raise
                frame, summary, paired = jw_tables(results, suite_root)
                print(f"Сохранено {len(results)}/{2*len(cases)*len(seeds)} запусков")
    jw_pair_plot(paired, suite_root, live)
    if live:
        display(summary.round(5))
        display(paired[["case", "seed", "risk_gain", "direct_gain", "joint_error_reduction"]].round(5))
    print("Файлы:", suite_root)
    return frame, summary, paired, suite_root


_jw_devices = ["cpu"]
if torch.cuda.is_available():
    _jw_devices.insert(0, "cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    _jw_devices.insert(0, "mps")
_jw_default = getattr(globals().get("device_ui"), "value", _jw_devices[0])
if _jw_default not in _jw_devices:
    _jw_default = _jw_devices[0]
jw_device_ui = widgets.Dropdown(options=_jw_devices, value=_jw_default, description="Устройство")
jw_run_button = widgets.Button(description="Запустить / продолжить", button_style="primary",
                                layout=widgets.Layout(width="230px"))
jw_zip_button = widgets.Button(description="Скачать ZIP")
jw_output = widgets.Output()


def _jw_start(_):
    global JW_RESULTS, JW_SUMMARY, JW_PAIRED, JW_RUN_ROOT
    jw_run_button.disabled = jw_zip_button.disabled = True
    with jw_output:
        jw_output.clear_output(wait=True)
        try:
            device = torch.device(jw_device_ui.value)
            if device.type == "cpu":
                torch.set_num_threads(min(4, os.cpu_count() or 1))
            JW_RESULTS, JW_SUMMARY, JW_PAIRED, JW_RUN_ROOT = jw_run_suite(
                JW_CONFIG, JW_CASES, JW_SEEDS, Path(OUTPUT_ROOT)/"joint_weight_ablation",
                device, live=True)
        finally:
            jw_run_button.disabled = jw_zip_button.disabled = False


def _jw_zip(_):
    with jw_output:
        root = Path(OUTPUT_ROOT)/"joint_weight_ablation"
        if not root.exists():
            print("Сначала запустите эксперимент.")
            return
        # ZIP включает и частично завершённые серии, в том числе checkpoints.
        archive = shutil.make_archive(str(root.parent/"joint_weight_ablation_results"),
                                      "zip", root)
        if globals().get("IN_COLAB", False):
            from google.colab import files
            files.download(archive)
        else:
            display(FileLink(archive))


jw_run_button.on_click(_jw_start)
jw_zip_button.on_click(_jw_zip)
print("balanced | shared | веса 1 и 2 | seed 5–9 | 15 000 train | 8 000 шагов")
print("После выполнения: summary.csv, paired.csv, results.csv и paired_comparison.png.")
print("risk_gain > 0 означает, что вес 2 уменьшил абсолютную ошибку Ω на этом seed.")
display(widgets.VBox([jw_device_ui, widgets.HBox([jw_run_button, jw_zip_button]), jw_output]))
