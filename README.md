from cgan_generator.cdcgan import fake_labels

# Material Generator with Generative AI
## cgan_generator
* **cgan_generator** proposes artificial materials based on the trained neural networks, using a conditional generative adversarial network (CGAN).
 
### Usage
```python
from ase.build import fcc111
from cgan_generator import train_and_generate
from cgan_generator import make_random_replacement
from cgan_generator import convert_to_dataframe

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