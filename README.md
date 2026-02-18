# paper_HighDimFECMC

This repository is a companion repository for the paper:

> **Diffusive Scaling Limits of Forward Event-Chain Monte Carlo: Provably Efficient Exploration with Partial Refreshment**

It contains Julia scripts that generate the Monte Carlo data and produce the plots used in Section 5 of the paper.

## Requirements

- Julia **1.12** (see `Project.toml`)

## Installation

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick run (short mode)

Most scripts support a short run mode controlled by environment variables. For example:

```bash
CI=true julia --project=. Section5-1/2-Logistic.jl
```

This is intended for quick sanity checks (e.g. in CI) and does **not** reproduce paper-quality runs.

## Reproducing the plots (Section 5)

Each figure is produced in two steps:

- **Step 1 (data)**: run the corresponding `*.jl` script to generate a `.jld2` file under `Section5-*/Data/`.
- **Step 2 (plot)**: run the corresponding `*-plot_*.jl` script to load the `.jld2` file(s) and create the figure.

### Section5-1

- **Standard Gaussian (ESS)**:
  - Data: `Section5-1/1-StandardGauss.jl` → `Section5-1/Data/1-StandardGauss.jld2`
  - Plot: `Section5-1/1-plot_StandardGauss.jl`
- **Standard Gaussian (ESS / CPU time)**:
  - Data: `Section5-1/1-StandardGauss.jl` → `Section5-1/Data/1-StandardGauss.jld2`
  - Plot: `Section5-1/1-plot_StandardGauss_CPUTime.jl`
- **Logistic target**:
  - Data: `Section5-1/2-Logistic.jl` → `Section5-1/Data/2-Logistic.jld2`
  - Plot: `Section5-1/2-plot_Logistic.jl`
- **Anisotropic Gaussian target (compound symmetry)**:
  - Data: `Section5-1/3-AnisoGauss.jl` → `Section5-1/Data/3-AnisoGauss.jld2`
  - Plot: `Section5-1/3-plot_AnisoGauss.jl`

### Section5-2

- **Anisotropic Gaussian target (varying correlation)**:
  - Data: `Section5-2/1-AnisoGauss.jl` → `Section5-2/Data/1-AnisoGauss.jld2`
  - Plot: `Section5-2/1-plot_AnisoGauss.jl`
- **Student target**:
  - Data: `Section5-2/2-Student.jl` → `Section5-2/Data/2-Student.jld2`
  - Plot: `Section5-2/2-plot_Student.jl`

### Section5-3

- **Batch means (Standard Gaussian target)**:
  - Data: `Section5-3/1-StandardGauss.jl` → `Section5-3/Data/1-StandardGauss.jld2`
  - Plot: `Section5-3/1-plot_SG.jl`
- **Batch means (Anisotropic Gaussian target)**:
  - Data: `Section5-3/2-AnisoGauss.jl` → `Section5-3/Data/2-AnisoGauss.jld2`
  - Plot: `Section5-3/2-plot_AG.jl`

Note: `Section5-3/2-plot_AG.jl` also reads `Section5-1/Data/3-AnisoGauss.jld2` to obtain a reference scale.

## Notes on runtime

Some scripts are configured with "paper setting" parameters (e.g. large `iter`) and can take hours.
If you only want to check that things run end-to-end, use the short mode described above.

## Citation

If you use this repository, please cite the accompanying paper. (Add arXiv/DOI information here.)