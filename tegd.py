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


# --- Step 2. Make the descriptor-target pair, by setting atomic numbers as descriptors and adsorption energy as target.

def make_samples(atomic_numbers, e_ads):
    samples = []
    for atomic_number, e_ad in zip(atomic_numbers, e_ads):
        sample = {
            'atomic_numbers': atomic_number,
            'adsorption_energy': e_ad
        }
        samples.append(sample)
    return samples

# テスト用のデータを定義
atomic_numbers = [
    [78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78],
    [78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 77],
    [78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 79]
]
e_ads = [-1.23, -1.45,-1.36]

def assign_rank(samples):
    # sort samples by adsorption energy
    samples = sorted(samples, key=lambda x: x['adsorption_energy'])
    # assign rank
    for i, sample in enumerate(samples):
        sample['rank'] = i + 1
    return samples

# samplesを生成
samples = make_samples(atomic_numbers, e_ads)
print(samples)
samples = assign_rank(samples)
print(samples)

# --- Step 3. Assign the strongest adsorption energy surfaces as "rank=1".
# samples = assign_rank(samples)

# --- Step 4. Put above dataset and train the CGAN.
# train_cgan()

# --- Step 5. Generate fake-samples for "rank=1" surfaces, to generate the surfaces with stronger adsorption energy.
# fake_samples = generate_by_cgan()