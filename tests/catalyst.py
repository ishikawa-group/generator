# A script for predict the adsorption energy with GAN.
# Step 1. Prepare the dataset by generating the surface structures and calculating the adsorption energy.
# Step 2. Make the descriptor-target pair, by setting atomic numbers as descriptors and adsorption energy as target.
# Step 3. Assign the strongest adsorption energy surfaces as "rank=1".
# Step 4. Put above dataset and train the CGAN.
# Step 5. Generate fake-samples for "rank=1" surfaces, to generate the surfaces with stronger adsorption energy.

num_samples = 10
e_ads = []
atomic_numbers = []

# --- Step 1. Prepare the dataset by generating the surface structures and calculating the adsorption energy.

for i in range(num_samples):
    surf = fcc111('Pt', size=(2, 2, 3), vacuum=10.0)
    e_ad = get_adsorption_energy(surface=surf)  # function to be implemented
    atomic_numbers.append(surf.get_atomic_numbers())  # get element information for surface

    print(f"Adsorption energy of sample {i+1}: {np.random.normal(0, 1)}")
    e_ads.append(e_ad)

# --- Step 2. Make the descriptor-target pair, by setting atomic numbers as descriptors and adsorption energy as target.
# samples = make_samples(atomic_numbers, e_ads)

# --- Step 3. Assign the strongest adsorption energy surfaces as "rank=1".
# samples = assign_rank(samples)

# --- Step 4. Put above dataset and train the CGAN.
# train_cgan()

# --- Step 5. Generate fake-samples for "rank=1" surfaces, to generate the surfaces with stronger adsorption energy.
# fake_samples = generate_by_cgan()
