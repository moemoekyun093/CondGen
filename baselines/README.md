# Upstream baselines

The baseline implementations in this directory are pinned Git submodules. Keep
project-specific adapters, data conversion, and Slurm wrappers outside these
directories so the upstream implementations remain unchanged.

| Baseline | Upstream repository | Pinned revision |
| --- | --- | --- |
| HARPOON | https://github.com/adis98/Harpoon | See the parent repository's submodule pointer |
| DiffPuter | https://github.com/hengruizhang98/DiffPuter | `2fa55373655b9e910146d94820fc1012da0dfd75` |
| GReaT | https://github.com/kathrinse/be_great | tag `0.0.9` (`d5689d485736780d11ecb0fbca24cb5255a9c61e`) |

GReaT is pinned to `0.0.9` because that is the version selected by HARPOON's
`be_great~=0.0.9` dependency. DiffPuter is pinned to the current revision of
its official ICLR 2025 repository at the time it was added.

After cloning the parent repository, initialize all baseline code with:

```bash
git submodule update --init --recursive
```
