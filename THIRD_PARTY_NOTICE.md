# Third-Party Notice

OpenPatch-PTW is designed as an academic extension of **GenPTW: Latent Image Watermarking for Provenance Tracing and Tamper Localization**.

Upstream project:

- Repository: `GanZhenliang/GenPTW`
- Paper: Zhenliang Gan, Chunya Liu, Yichao Tang, Binghao Wang, Shiwen Cui, Weiqiang Wang, Xinpeng Zhang. *GenPTW: Latent Image Watermarking for Provenance Tracing and Tamper Localization*. AAAI 2026.
- Upstream license: Non-Commercial Research License.

This repository intentionally does not vendor the full GenPTW source tree or its model weights. `scripts/bootstrap_genptw.sh` clones the upstream repository into `third_party/GenPTW`, and users must follow the upstream license and model/data licenses.

Suggested citation:

```bibtex
@inproceedings{gan2026genptw,
  title={GenPTW: Latent Image Watermarking for Provenance Tracing and Tamper Localization},
  author={Gan, Zhenliang and Liu, Chunya and Tang, Yichao and Wang, Binghao and Cui, Shiwen and Wang, Weiqiang and Zhang, Xinpeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={5},
  pages={4085--4093},
  year={2026}
}
```
