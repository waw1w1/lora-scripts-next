# Review fixes (2026-09-23)

REV-01 through REV-05 and TEST-01 are addressed; see
`docs/qwen-edit-review-fixes.md` for implementation and evidence boundaries.
Edit remains experimental: TEST-02 GPU, native Windows and real ComfyUI output
acceptance are still outstanding. The older baseline Processor test below has
been corrected to match managed Processor preparation, without production changes.

# Edit adaptation validation

The `qwen2.1-image-edit` branch adds Edit to the existing Qwen 2.1 schema and
DiffSynth entry; the older report below describes the original T2I revision.

- Uses the same pinned upstream commit and official `extra_inputs=edit_image`.
- Reuses reference-path, preview, config, task and dry-run interfaces.
- New CPU tests cover multi-reference metadata/folder pairing, config round-trip,
  cache reuse/invalidation, conditioned positive/negative preview cache, real
  upstream parser/dataset/training-input/model-function contracts, and dry-run HTTP.
  Cache encoder tests use synthetic stand-ins, not the real 8B encoder or VAE.
- Frontend checks cover mode buttons, draft persistence, conditional reference
  fields, T2I serialization and existing shared controls.
- Edit batch size is explicitly limited to 1. Gradient accumulation is supported.
- No real GPU training, trained LoRA quality, or ComfyUI image-generation acceptance
  was performed. The unchanged dev base already fails
  `test_comfy_component_mode_and_processor_errors`: it expects local Processor
  validation despite the newer managed Processor preparation path.

---

# DiffSynth review validation

This revision supersedes the original PR smoke report. Only Qwen-Image-2.1 BF16
t2i LoRA is exposed. GPU training is deliberately not run in this environment.

## Sources checked

- GUI dev base: `b4df253cfc5d2d6901651c15be909cbd10bd1118`.
- Complete DiffSynth checkout: `7686e54d41d25c0e8ed5f1318acc23b6bb832654`, unchanged.
- ComfyUI loader: `b0f4b7b294ce482a2e071d9d762c133d38c7aa07`, unchanged.
- Comfy-Org/Qwen-Image-2.1 BF16 safetensors headers (no real tensors downloaded).
- PreviewSampleField interaction reused from author's `feat/ai-toolkit-klein` branch,
  with edit-only/unsupported controls removed for this t2i entry.

## Verified boundaries

- Full BF16 component key/shape normalization matches all three pinned DiffSynth
  model hashes. Tiny tensor tests verify fused MLP row ordering and VAE time-axis
  removal; cache writes do not modify input files.
- Real PEFT-generated synthetic LoRA -> safetensors -> unmodified ComfyUI
  `model_lora_keys_unet`, `load_lora`, `calculate_weight`: all tensors mapped,
  both fused MLP halves and ordinary output layer produce the expected numerical
  weight update. This is not full-model inference or a training-quality test.
- Mixed image/TXT directory repeats, ordinary/root images, empty/missing captions,
  original CSV/JSON/JSONL metadata, mode-specific config import/export.
- Structured preview samples, optimizer-update accounting including a final partial
  accumulation, scheduler/RNG/mode restoration, existing task preview API filenames.
- Real installer supervisor and child process cancellation through existing Task;
  environment lock release; ready invalidation on source change.
- HTTP `/api/run` -> Task -> actual Accelerate subprocess -> thin entry -> actual
  pinned upstream parser: exit 0 with submitted parameters. CUDA readiness alone
  is substituted in the test process; `--check-only` prevents tensor loading and
  training. Production installation still requires a working CUDA runtime.
- Backend regression: 139 tests passed, including engine dispatch, config import/export,
  task lifecycle/insights, SPA routes and the DiffSynth/Comfy tests.
- Frontend Node 22: typecheck, lint, 211 Vitest tests and production build passed.
  Existing lint warnings in EngineStatusBar and the bundle-size warning remain.

## Remaining acceptance

- Native Windows installation/cancellation and real RTX 5090 D GPU training have
  not been repeated for this revision.
- Actual exported trained Qwen LoRA must still be reloaded and sampled with native
  ComfyUI nodes; the CPU synthetic loader test is not a substitute.
- The cloud browser rejects this environment's localhost (`ERR_BLOCKED_BY_CLIENT`)
  and no supported preview address is available. The real backend runs locally;
  this revision has component tests and HTTP/subprocess smoke, not browser-click
  end-to-end evidence or new screenshots.
- CPU model offload + in-training sampling is explicitly rejected. The pinned
  upstream runner does not expose its OffloadTrainingManager to logger callbacks.
  Supporting that combination requires a public upstream lifecycle hook; do not
  claim it works, copy the training loop or reach into private hook closures.

See `docs/diffsynth.md` for user-facing setup, Processor requirements and limits.
