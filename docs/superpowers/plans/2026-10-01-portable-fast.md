# GUI + Fast Portable Implementation

- [x] Regression tests for relocation, no-op on developer venv, missing base.
- [x] Marked runtime repair before Fast audit/discovery.
- [x] Profile-aware launcher without host torch or Kohya setup.
- [x] Install declared GUI dependencies after temporary prefetch cleanup.
- [x] Include psutil required directly by GUI task management.
- [x] Bundle the Fast base CPython without its global site-packages.
- [x] Validate runtime paths and run strict engine audit before archiving.
- [x] Remove host torch installation from Fast GUI maintenance helper.
- [ ] Fresh local build and moved/extracted-package validation.
- [ ] Joint updater tests, GUI restart and short GPU training.

No artifact upload, release publication or remote merge is authorized here.
The available source Fast venv uses torch 2.11/cu130 while current dev expects
2.12/cu132. Do not weaken this audit to make a historical runtime pass.
