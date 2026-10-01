# GUI + Fast Portable Implementation

- [x] Regression tests for relocation, no-op on developer venv, missing base.
- [x] Marked runtime repair before Fast audit/discovery.
- [x] Profile-aware launcher without host torch or Kohya setup.
- [x] Install declared GUI dependencies after temporary prefetch cleanup.
- [x] Include psutil required directly by GUI task management.
- [x] Bundle the Fast base CPython without its global site-packages.
- [x] Validate runtime paths and run strict engine audit before archiving.
- [x] Remove host torch installation from Fast GUI maintenance helper.
- [x] Fresh local build and moved/extracted-package validation.
- [x] Joint updater tests, GUI restart and short GPU training.

No artifact upload, release publication or remote merge is authorized here.
The available source Fast venv uses torch 2.11/cu130 while current dev expects
2.12/cu132. Do not weaken this audit to make a historical runtime pass.

## Local Acceptance (2026-10-01)

The dedicated test runtime was refreshed with the maintained installer and
passed the strict cu132 audit. No shared environment was modified.
Both PRs together passed 131 tests and 2 subtests. The real builder passed
its 34 embedded Git tests (1 skipped).

A local 7z archive (3,205,672,030 bytes) was extracted into a path containing
spaces. Embedded host/Fast runtime paths, CUDA and strict engine audit passed.
Offline GUI startup returned HTTP 200 before and after the actual root BAT
updater. A local-only remote supplied the update: an untracked-file conflict
blocked safely, then normal update, launcher self-replacement and repeat update
passed. Six userdata fixtures and six selected runtime files retained their
SHA256 hashes; tracked files remained clean and no stash was created.

The updated extracted package completed an Anima 2.9B, AdamW, bf16, rank-4,
3-step GPU smoke on an RTX 4090 and saved a 41,152,704-byte LoRA.
This is not long-run training-quality validation. External model weights were
read-only; the host does not include torch or transformers. Marketplace,
optional tagger downloads and public-network update fallbacks were not covered.
No artifacts were uploaded and neither PR was merged.
