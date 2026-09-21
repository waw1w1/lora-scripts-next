# DiffSynth-Studio integration notes

## Scope

Only Qwen-Image-2.1 text-to-image LoRA, single NVIDIA GPU, BF16. The complete
upstream checkout is pinned at `7686e54d41d25c0e8ed5f1318acc23b6bb832654` and is
not patched. Image editing, full finetuning, other models, multi-GPU and
in-training sampling are not exposed.

The engine owns a uv-managed Python 3.12 under `extensions/diffsynth/.python`
and packages under `extensions/diffsynth/.venv`. Installation sets
`UV_PYTHON_INSTALL_DIR` to the engine directory; repair recreates both.
The GUI does not import DiffSynth or share its Python packages. PyTorch
2.8.0/cu128 and torchvision 0.23.0 are installed first; the training environment
also installs Transformers 4.57.x, TensorBoard and bitsandbytes. No DeepSpeed,
FlashAttention or Bash launcher is required.

## Validation matrix

| Environment | Python | torch | CUDA / GPU | Result | Date |
| --- | --- | --- | --- | --- | --- |
| Linux x86_64 cloud, independent interpreter | 3.12.14 | 2.8.0+cu128 | CUDA 12.8 package; no GPU attached | Source checkout, dependency install, pipeline imports and official `train.py --help` passed. Production readiness correctly rejected missing CUDA GPU. | 2026-09-21 |
| Windows 11 / RTX 5090 D | Not run | Not run | Not available in this environment | Requires target-machine verification; no training or VRAM claim. | 2026-09-21 |

## Integration details / findings

- P2: use base DiffSynth dependencies, TensorBoard and the selected NF4 backend;
  do not install the optional `[training]` extra that brings DeepSpeed.
- P3: launch through the independent Python, remove inherited PYTHONPATH and
  disable user-site packages. Pass argv as a list and resolve filesystem paths
  before changing cwd. Chinese paths, spaces and an empty target-layer argument
  are covered by tests.
- P6: use `diffsynth_model_dir` for the model root instead of borrowing Kohya's
  base-model field.
- P7: keep the complete upstream Git checkout, including examples and root files.
- P8: register official Qwen assets with the existing download UI and require
  local transformer/text_encoder/vae/processor files. Check shard indexes and
  processor/template files before submission. Training uses local model paths
  and HF offline mode; it does not initiate an asset download.
- Upstream `DataLoader` uses a lambda collator. Workers are fixed at zero to
  avoid Windows spawn pickling failures, without patching upstream.
- Upstream empty `lora_target_modules` enables automatic target detection;
  preserve the empty argv value rather than falling back to the generic parser
  default `q,k,v,...`.
- UI TOML is separate from native argument JSON and dataset JSON. TensorBoard
  lives in the run's `tensorboard_log` directory, using the `loss` tag. The
  shared insights resolver must take the actual runtime directories from task
  metadata (the UI TOML contains the parent output directory).
- Each submission receives a separate output subdirectory. This prevents older
  appended CSV rows and TensorBoard events being mistaken for the new run.
- CSV `step` counts batches, as does the upstream save interval. It is not the
  optimizer-update count when gradient accumulation is enabled.

## Smoke evidence

| Qwen-Image-2.1 LoRA check | Result |
| --- | --- |
| Install complete source + managed Python + venv + dependencies | Passed; CUDA readiness check blocked on CPU-only host as expected |
| Official training script imports and argument help | Passed without model loading |
| Real engine-management page and model/engine selector | Visible in Chromium; unavailable runtime correctly gates training |
| Real form -> HTTP `/api/run` -> task manager -> Popen | Passed using test-only CUDA readiness fixtures |
| Actual isolated Accelerate -> pinned official `train.py --help` | Exit 0; no training executed |
| LoRA weights / sample images / GPU memory | Not tested; user explicitly requested no training |

The browser smoke changes **only its test server process**: CUDA readiness is
mocked and `--help` is appended at the task execution boundary. Production code
has no bypass flag. Placeholder model files are only preflight fixtures and
are never loaded. This proves transport/launch, not successful GPU training.

Local smoke evidence: `.runtime/diffsynth-smoke/{submission,launch,task}.json`,
`result.log`, `production-server.log`, `test-server.log`, and UI screenshots.
Test assets and environments are excluded from Git.

## Dev branch validation

The adaptation is based on `origin/dev` at
`b4df253cfc5d2d6901651c15be909cbd10bd1118`, not the main release branch.
Only DiffSynth changes were transferred; dev's updater, homepage, version and
dataset-config behavior remain intact. Frontend assets were rebuilt from dev.

Regression checks: `frontend/npm run check` (209 tests plus typecheck/lint/build);
114 targeted Python tests passed, 3 existing torch-dependent checks skipped in
the lightweight GUI test environment, plus 6 passing subtests. Scope includes
engine dispatch, adapters, config import/export, asset checks, task insights,
SPA routes and dev dataset-config/image-scan/preview regressions.
See `tests/test_diffsynth_engine.py`.

The full repair API was rerun on dev: source download, local Python installation,
venv creation, torch/dependency installation and official entry imports passed.
`sys.base_prefix` resolves inside the engine's `.python` directory. The only
audit error is the expected missing CUDA GPU. Recreating the venv uses
`uv venv --clear` so a partial environment cannot block an installation retry.
Evidence: `.runtime/dev-validation/install-result.json` and `install.log`.

The browser smoke was rerun against the dev build: the real production gate
and engine selector render correctly; the test server's frontend POST reaches
the real task process and isolated official `train.py --help` exits 0, with
no browser errors. Same test-only CUDA/--help boundaries described above apply.
Evidence: `.runtime/dev-validation/browser.log` and the refreshed smoke JSONs.
