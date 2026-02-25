# MT-8: Coverage-Aware Eviction Analysis

| Domain | Total Protos | Zero-Load | Fragility % | Gini | LFU-vs-Coverage Spearman | Policy Conflicts |
|--------|-------------|-----------|-------------|------|--------------------------|-----------------|
| Dogs | 10279 | 5771 (56.1%) | 5.0% | 0.710 | 0.814 | 42 |
| Birds | 5773 | 2845 (49.3%) | 6.3% | 0.663 | 0.843 | 48 |
| Cars | 7882 | 3645 (46.2%) | 11.5% | 0.635 | 0.711 | 77 |
| Aircraft | 3276 | 1430 (43.7%) | 32.5% | 0.608 | -0.151 | 15 |


### Analysis

LFU tracks write-path access (nearest neighbor during training), while Irreplaceability tracks read-path coverage (sole same-class nearest neighbor during inference). The Gini coefficient measures the inequality of prototype loads, and the Spearman correlation shows how well LFU ranks align with coverage loss.
