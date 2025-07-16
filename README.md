# Delta-learning-force

This project applies delta-learning to predict the difference between force field and DFT-level interaction energies for water-ion and water-water dimers.

**Why it matters:** Classical force fields fail in reactive/interfacial systems. This model learns quantum corrections, enabling high accuracy at low cost.

## Workflow:
- Parse xyz file, generate dimer sets
- Predict delta energies using PyTorch feedforward NN
- Combine with force field baseline
- Evaluate with MAE / Pearson metrics

## Results:
- MAE: 0.28 kcal/mol (test set)
- Pearson r: 0.95
- Supports F⁻, Cl⁻, Br⁻, I⁻, K⁺, Na⁺, Mg²⁺, Ca²⁺    clusters

## How to Run:
python predict.py input.xyz model.pt
