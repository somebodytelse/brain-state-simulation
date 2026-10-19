# Contributing

1. Create a Python environment and install `python -m pip install -e .`.
2. Run `python -m unittest discover -s tests -v`.
3. Put reusable functions under `src/brain_states`; put run orchestration under
   `src/brain_states/experiments` and rendering helpers under `visualization`.
4. Keep signals, inferred activity and ground-truth evaluation separate.
5. Document shapes, units, seeds and changed parameters. A method change belongs
   in a new experiment, not a silent modification to historical results.
6. Never tune detector thresholds using truth or infer success only from
   reconstruction. Report unmatched and extra states.
7. Keep old root launchers working when moving modules. Test both installed
   commands and the source-checkout entry points.
8. Do not commit environments, temporary output, secrets or regenerated large
   datasets. Add small useful reference assets deliberately.

The frozen original `state_detection.py` implementation is fingerprinted by
historical experiments. Moving it should preserve its bytes; intentional future
scientific edits require an explicit versioned fingerprint and new reference
results, not changing a hash to hide a numerical difference.

The historical three-state clustering example is preserved under `legacy/`,
requires the optional legacy dependencies, and is not part of test discovery.
A project license has not yet been chosen by the owner.
