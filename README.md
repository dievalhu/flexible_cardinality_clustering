# Semi-Supervised Soft Clustering with Flexible Cardinality
## Overview

This repository contains the implementation of **CapFlex**, a semi-supervised clustering framework proposed in the paper *"Semi-Supervised Soft Clustering with Flexible Cardinality"*.

Clustering under size requirements is critical in operational settings (e.g., workload distribution, territorial partitioning) but is often hindered by rigid constraints that overrule data similarity. **CapFlex** treats target cluster sizes as *soft* requirements by allowing bounded deviations around an ideal cardinality vector.

Key features:
* **Flexible Constraints:** Allows defined tolerance margins ($\delta$) around target sizes instead of strict equality.
* **Hybrid Optimization:** Combines stochastic exploration of feasible cardinality vectors with an exact Mixed-Integer Linear Programming (MILP) assignment step.
* **Pareto Optimization:** Balances the trade-off between structural quality (Silhouette) and cardinality compliance.

## Methodology
The proposed approach uses a two-level algorithm. The outer level performs a bounded exploration of the cardinality search space (using Random Search), while the inner level solves the optimal assignment problem for each candidate capacity vector using MILP.

![CapFlex Architecture](img/methodology.png)
*Figure 1: The CapFlex framework architecture, illustrating the coupling of stochastic capacity exploration with MILP-based assignment optimization.*

## Installation
This project is written in **R**. To replicate the experiments, you need a working R environment (version 4.0.0 or higher is recommended).

### Prerequisites

* **R**: Download from [CRAN](https://cran.r-project.org/).
* **RStudio** (Optional but recommended).

### Dependencies
```r
install.packages(c(
  "foreach",     
  "doParallel",  
  "lpSolve",     
  "readr",       
  "dplyr",       
  "aricode",     
  "cluster"
```

## Usage

### 1. Running the Analysis

The main script is designed to be executed either from the command line or interactively within RStudio.

**Option A: Command Line**

To run the complete pipeline using `Rscript`:

```bash
Rscript CapFlex.R
```

**Option B: RStudio**

1. Open the project in RStudio.
2. Open the `CapFlex.R` script.
3. Adjust the parameters at the beginning of the file to select the dataset or tolerance.
4. Click the **Source** button to run the complete pipeline.
