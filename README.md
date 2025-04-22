# Material Generator with Generative AI

A Python package for generating artificial materials using conditional generative adversarial networks (CGAN).

## Installation

```bash
pip install .
```

## Requirements

* Python >= 3.10
* PyTorch >= 2.3.0
* PyTorch Lightning >= 2.5.0
* ASE >= 3.24.0
* DGL >= 2.2.1
* PyMatGen >= 2025.2.18
* Other dependencies are automatically installed

## Test code
```python
from generator.cgan_generator import gazouno_cgan

target_figure = "3"
gazouno_cgan(target_figure)  # -> "3"の手描き文字の画像が生成される
```

## Usage
```python
from ase.build import fcc111
from generator.cgan_generator import (
    train_and_generate,
    make_random_replacement,
    convert_to_dataframe
)

# make initial samples
samples = []
for i in range(10):
    surf = fcc111("Pt", size=(3, 3, 2))
    surf = make_random_replacement(surf)
    samples.append(surf)

# calculate reactions energies
e_rxns = []
for isample in samples:
    e_rxn = isample.get_reaction_energy()
    e_rxns.append(e_rxn)

df = convert_to_dataframe(samples, e_rxns)

# generate fake samples
fake_samples = train_and_generate(df)

# calculate reaction energies for generated samples
e_rxns_new = []
for isample in fake_samples:
    e_rxn = isample.get_reaction_energy()
    e_rxns_new.append(e_rxn)
```
