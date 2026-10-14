"""Code-level parity of scripts/run_retrain_209.py with the PR #1 (run_sanity.py, ResNet50) and PR #2 (run_multimodel.py) training code.
This is TEXTUAL/STRUCTURAL verification only. It is not a numerical GPU validation: nothing is trained or forwarded here."""
import re
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
norm = lambda s: re.sub(r"\s+", " ", s.strip())
lines = lambda f: {norm(l) for l in (ROOT / "scripts" / f).read_text().splitlines()}
NEW, PR2, PR1 = lines("run_retrain_209.py"), lines("run_multimodel.py"), lines("run_sanity.py")

TRAIN = ['torch.manual_seed(CFG["train_seed"])', 'np.random.seed(CFG["train_seed"])',
         'opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])',
         'steps = CFG["epochs"] * int(np.ceil(n / CFG["batch_size"]))',
         'sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)',
         'g_data = torch.Generator().manual_seed(CFG["train_seed"])', 'g_aug = torch.Generator().manual_seed(CFG["train_seed"] + 1)',
         'for ep in range(CFG["epochs"]):', 'model.train()', 'perm = torch.randperm(n, generator=g_data)',
         'for b in range(0, n, CFG["batch_size"]):', 'bi = perm[b:b + CFG["batch_size"]]', 'xb = augment(Xt[tr_idx[bi.numpy()]], g_aug)',
         'yb = ytr[bi].to("mps")', 'logits = model(to_input(xb).to("mps"))', 'loss = F.cross_entropy(logits, yb)',
         'opt.zero_grad(set_to_none=True)', 'loss.backward()', 'opt.step()', 'sched.step()', 'tl += loss.item() * len(bi)',
         'tc += (logits.argmax(1) == yb).sum().item()', 'tr_idx = np.array([pos[int(i)] for i in tr.image_id])',
         'te_idx = np.array([pos[int(i)] for i in te.image_id])', 'ytr = torch.tensor(tr.label.to_numpy() - 1)', 'assert n % CFG["batch_size"] != 1',
         'model.eval()', 'with torch.no_grad():', 'for b in range(0, len(te_idx), CFG["eval_batch_size"]):',
         'outputs.append(model(to_input(Xt[te_idx[b:b + CFG["eval_batch_size"]]]).to("mps")).float().cpu())',
         'lg = torch.cat(outputs).numpy().astype(np.float64)', 'assert np.isfinite(lg).all()', 'Xt = torch.from_numpy(X)',
         'ids, X = load_all(ROOT / "reports/manifests/image_manifest.csv")', 'assert len(ids) == 3064 and X.shape == (3064, 224, 224)']


@pytest.mark.parametrize("stmt", TRAIN)
def test_training_statement_identical_in_pr2_and_new_runner(stmt):
    assert stmt in PR2, f"not in PR2 run_multimodel.py: {stmt}"
    assert stmt in NEW, f"missing/changed in run_retrain_209.py: {stmt}"


@pytest.mark.parametrize("stmt", ['model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)', 'model.fc = torch.nn.Linear(2048, 3)',
                                  'torch.manual_seed(CFG["train_seed"])', 'opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])'])
def test_resnet50_statements_identical_to_pr1(stmt):
    assert stmt in norm((ROOT / 'scripts/run_sanity.py').read_text()) and stmt in NEW  # PR#1 joins some statements with ';' on one line


def test_no_private_hyperparameters_and_cfg_imported():
    src = (ROOT / "scripts/run_retrain_209.py").read_text()
    assert "from run_sanity import CFG, augment, to_input" in src
    assert not re.search(r"^\s*CFG\s*=", src, re.M) and not re.search(r"\blr\s*=\s*[0-9]", src) and "early_stop" not in src and "class_weight" not in src
    assert "from run_multimodel import make_model" in src and "from leakage.preprocess import load_all" in src


def test_frozen_cfg_values():
    from run_sanity import CFG
    assert (CFG["batch_size"], CFG["epochs"], CFG["lr"], CFG["weight_decay"], CFG["train_seed"], CFG["eval_batch_size"], CFG["optimizer"], CFG["input"]) == (32, 10, 1e-4, 1e-4, 0, 64, "AdamW", 224)
    assert CFG["early_stopping"] is False and CFG["model_selection"].startswith("none") and "cosine" in CFG["schedule"] and "hflip p=0.5" in CFG["augmentation"]
    assert "ImageNet mean/std" in CFG["normalisation"] and "no class weights" in CFG["loss"]


def test_architectures_and_weights_match_originals(monkeypatch):
    import torch, torchvision
    import run_multimodel as PR2M
    import run_retrain_209 as R
    for n in ("mobilenet_v2", "efficientnet_b0", "densenet121"):
        assert "IMAGENET1K_V1" in str(PR2M.WEIGHTS[n]); monkeypatch.setitem(PR2M.WEIGHTS, n, None)  # structure only; no weight download, no forward pass
    seen = {}
    orig = torchvision.models.resnet50
    monkeypatch.setattr(torchvision.models, "resnet50", lambda weights=None: (seen.__setitem__("w", weights), orig(weights=None))[1])
    heads = {"resnet50": lambda m: (m.fc.in_features, m.fc.out_features), "mobilenet_v2": lambda m: (m.classifier[1].in_features, m.classifier[1].out_features),
             "efficientnet_b0": lambda m: (m.classifier[1].in_features, m.classifier[1].out_features), "densenet121": lambda m: (m.classifier.in_features, m.classifier.out_features)}
    want = {"resnet50": (2048, 3), "mobilenet_v2": (1280, 3), "efficientnet_b0": (1280, 3), "densenet121": (1024, 3)}
    for n in R.MODELS:
        assert heads[n](R.build_model(n)) == want[n]
    assert seen["w"] == torchvision.models.ResNet50_Weights.IMAGENET1K_V1
