#!/usr/bin/env bash
# Prepare a GPU VM. Installs software and verifies the GPU path with a seconds-long check.
# Downloads no datasets, uses no API key, and launches no experiment.
#
#   bash scripts/vm_setup.sh
#
# Set PYTHON to choose an interpreter (3.12+). If the default PyTorch build does not
# match the driver, set TORCH_INDEX_URL, e.g. https://download.pytorch.org/whl/cu128.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ is required"'
# A failed earlier attempt can leave a .venv without pip.
[ -x .venv/bin/python ] && .venv/bin/python -m pip --version >/dev/null 2>&1 || rm -rf .venv
[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip

if [ -n "${TORCH_INDEX_URL:-}" ]; then
  .venv/bin/python -m pip install --quiet --force-reinstall "torch==2.14.1" --index-url "$TORCH_INDEX_URL"
fi
# The lock pins versions; platform-specific dependencies (CUDA libraries) resolve here.
.venv/bin/python -m pip install --quiet -c requirements-lock.txt -e '.[all,dev]'

# The full suite, including real subprocess training on tiny fixtures, runs here and
# never on the project owner's laptop.
echo "== software tests, about 3 minutes =="
.venv/bin/python -m pytest -q

echo "== hardware =="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || {
  echo "nvidia-smi failed: no usable GPU driver." >&2
  exit 1
}
echo "CPUs: $(nproc)   Free disk here: $(df -h . | awk 'NR==2 {print $4}')"

echo "== GPU check: each model family and loss option, identical across two runs =="
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python - <<'PY'
import torch
import transformers

from rsi_ml.schemas import Recipe
from rsi_ml.tasks.cnn import CifarResNet

assert torch.cuda.is_available(), "PyTorch cannot see CUDA; set TORCH_INDEX_URL for this driver."
torch.use_deterministic_algorithms(True)
recipe = Recipe(
    hidden_sizes=[], activation="relu", normalization="none", dropout=0.0, optimizer="sgd",
    learning_rate=0.1, weight_decay=0.0005, batch_size=128, epochs=1, schedule="cosine",
    warmup_fraction=0.0, decay_fraction=0.2, momentum=0.9,
    cnn_channels=[64, 128, 256, 512], cnn_blocks=[2, 2, 2, 2],
)


def steps(model, optimizer, inputs, targets, classes, **loss_options):
    values = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = torch.nn.functional.cross_entropy(output, targets, **loss_options)
        loss.backward()
        optimizer.step()
        values.append(loss.item())
    return values


def resnet():
    torch.manual_seed(1)
    model = CifarResNet(100, recipe).cuda()
    x, y = torch.rand(128, 3, 32, 32).cuda(), torch.randint(0, 100, (128,)).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # Label smoothing and class weights are both choices agents can make.
    return steps(
        model, optimizer, x, y, 100, label_smoothing=0.1, weight=torch.ones(100).cuda()
    )


def bert():
    torch.manual_seed(1)
    config = transformers.BertConfig(
        vocab_size=1000, hidden_size=128, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=2, max_position_embeddings=512, hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    config._attn_implementation = "eager"
    encoder = transformers.BertModel(config, add_pooling_layer=False).cuda()
    head = torch.nn.Linear(128, 2).cuda()
    ids = torch.randint(0, 1000, (32, 128)).cuda()
    mask = torch.ones_like(ids)
    y = torch.randint(0, 2, (32,)).cuda()

    def model(inputs):
        return head(encoder(input_ids=inputs, attention_mask=mask).last_hidden_state[:, 0])

    parameters = [*encoder.parameters(), *head.parameters()]
    return steps(
        model, torch.optim.AdamW(parameters, lr=1e-4), ids, y, 2, weight=torch.ones(2).cuda()
    )


for name, run in (("ResNet-18", resnet), ("BERT", bert)):
    first, second = run(), run()
    assert first == second, f"{name} is not deterministic on this GPU: {first} != {second}"
    print(f"{name}: deterministic")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}")
PY
echo "Setup complete. Nothing was downloaded or launched."
