# Security Notes

## Dependency alerts

Dependency alerts are tracked through GitHub Dependabot for GitHub Actions and
Python packages. The Python environment is locked with `uv.lock`; remedial
updates should be made with:

```sh
uv lock --upgrade
uv lock --check
```

The current lockfile refresh updates or removes the vulnerable transitive
packages that can be remediated within the existing dependency constraints,
including patched versions of `aiohttp`, `onnx`, `pillow`, `setuptools`,
`torch`, `transformers`, and `urllib3`.

One known dependency alert remains constrained by an upstream package:
`hydra-core` is locked at `1.3.2`, while the patched advisory range starts at
`1.3.4`. The ASR runtime dependency currently requires `hydra-core >1.3, <=1.3.2`,
so adding `hydra-core>=1.3.4` makes the dependency graph unsatisfiable. Revisit
this when the upstream runtime relaxes its constraint or when the runtime path
can be made optional or replaced.

## Run artifacts

Private and ad-hoc run outputs must stay out of Git. They can contain real
transcripts or other sensitive benchmark material and are ignored by default
under `runs/*`.

Synthetic benchmark outputs are safe fixtures and may be committed under
`runs/synthetic*/`.