# A script for predict the adsorption energy with GAN.
# Step 1. Prepare the dataset by generating the surface structures and calculating the adsorption energy.
# Step 2. Make the descriptor-target pair, by setting atomic numbers as descriptors and adsorption energy as target.
# Step 3. Assign the strongest adsorption energy surfaces as "rank=1".
# Step 4. Put above dataset and train the CGAN.
# Step 5. Generate fake-samples for "rank=1" surfaces, to generate the surfaces with stronger adsorption energy.

from conditional_gan.get_adsorption_energy import get_adsorption_energy

num_samples = 4
e_ads = []
atomic_numbers = []

# --- Step 1. Prepare the dataset by generating the surface structures and calculating the adsorption energy.
from ase.build import fcc111
import numpy as np

for i in range(num_samples):
    surf = fcc111("Ni", size=(3, 3, 4), vacuum=10.0)
    # add some random replacement of surface atoms

    e_ad = get_adsorption_energy(surface=surf)  # function to be implemented
    atomic_numbers.append(surf.get_atomic_numbers())  # get element information for surface

    print(f"Adsorption energy of sample {i+1}: {np.random.normal(0, 1)}")
    e_ads.append(e_ad)

# --- Step 2. Make the descriptor-target pair, by setting atomic numbers as descriptors and adsorption energy as target.
# samples = make_samples(atomic_numbers, e_ads)
def make_samples(atomic_numbers, e_ads):
    samples = []
    for atomic_number, e_ad in zip(atomic_numbers, e_ads):
        sample = {
            'atomic_numbers': atomic_number,
            'adsorption_energy': e_ad
        }
        samples.append(sample)
    return samples

samples = make_samples(atomic_numbers,e_ads)


# --- Step 3. Assign the strongest adsorption energy surfaces as "rank=1".
# samples = assign_rank(samples)
def assign_rank(samples):
    # sort samples by adsorption energy
    samples = sorted(samples, key=lambda x: x['adsorption_energy'])
    # assign rank
    for i, sample in enumerate(samples):
        sample['rank'] = i + 1
    return samples

samples = assign_rank(samples)

print(samples)



# --- Step 4. Put above dataset and train the CGAN.
# train_cgan()

# --- Step 5. Generate fake-samples for "rank=1" surfaces, to generate the surfaces with stronger adsorption energy.
# fake_samples = generate_by_cgan()