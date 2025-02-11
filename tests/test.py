# A script for minimize the activation energy with GAN.
# Step 1. Prepare the dataset by generating the surface structures and calculating the activation energy.
# Step 2. Make the descriptor-target pair, by setting atomic numbers as descriptors and activation energy as target.
# Step 3. Assign the smallest activation energy surfaces as "rank=1".
# Step 4. Put above dataset and train the CGAN.
# Step 5. Generate fake-samples for "rank=1" surfaces, to generate the surfaces with lower activation energy.

import sys
sys.path.append("../")
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ase.build import fcc111
from conditional_gan.get_reaction_energy import get_reaction_energy

ATOMIC_NUMBERS = {"Ni": 28, "Pd": 46}


def make_samples(atomic_numbers, e_acts):
    # Make descriptor-target pairs
    samples = []
    for atom_num, e_acts in zip(atomic_numbers, e_acts):
        samples.append({
            "atomic_numbers": atom_num,
            "activation_energy": e_acts
        })

    # --- Normalize atomic numbers
    # Get the maximum number of atomic numbers
    max_of_max = np.max(np.array([np.max(samples[i]["atomic_numbers"]) for i in range(len(samples))]))
    for sample in samples:
        sample["atomic_numbers_scaled"] = sample["atomic_numbers"] / max_of_max

    return samples


def assign_rank(samples):
    # Assign ranks starting from 0
    samples = sorted(samples, key=lambda x: x["activation_energy"])

    n_rank = 5
    index = np.array_split(np.arange(len(samples)), n_rank)
    for i, idx in enumerate(index):
        for j in idx:
            samples[j]["rank"] = i

    return samples


class AtomicDataset(Dataset):
    """
    データセットクラス
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # atomic_numbers を特徴ベクトルに
        atomic_numbers_scaled = torch.tensor(self.samples[idx]["atomic_numbers_scaled"], dtype=torch.float32)
        energy = torch.tensor(self.samples[idx]["activation_energy"], dtype=torch.float32)
        rank = torch.tensor(self.samples[idx]["rank"], dtype=torch.long)
        return atomic_numbers_scaled, energy, rank


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_size + n_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        labels_onehot = eye[labels]
        x = torch.cat([x.view(x.size(0), -1), labels_onehot], dim=1)
        return self.net(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nz, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, image_size),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)


def make_noise(labels):
    # ラベル情報をノイズに加える
    labels_onehot = eye[labels]                        # (batch_size, n_classes)
    labels_onehot = labels_onehot.repeat_interleave(nz // n_classes, dim=-1)
    z = torch.normal(0, noise_std, size=(len(labels), nz), device=device)
    z = z + labels_onehot                              # (batch_size, nz)
    return z


def make_false_labels(labels):
    diff = torch.randint(1, n_classes, size=labels.size(), device=device)
    fake_labels = (labels + diff) % n_classes
    return fake_labels


def train(netD, netG, optimD, optimG, n_epochs):
    criterion = nn.BCELoss()
    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)
    netD.train()
    netG.train()
    for epoch in range(1, n_epochs+1):
        for atomic_numbers_scaled, _, ranks in dataloader:
            atomic_numbers_scaled = atomic_numbers_scaled.to(device)
            ranks = ranks.to(device)

            # --- Discriminator 学習 ---
            z = make_noise(ranks)
            fake_data = netG(z)
            pred_fake = netD(fake_data, ranks)       # 偽データ 判定
            pred_real = netD(atomic_numbers_scaled, ranks)  # 本物データ 判定

            lossD_fake = criterion(pred_fake, fake_labels)
            lossD_real = criterion(pred_real, real_labels)
            lossD = lossD_fake + lossD_real

            optimD.zero_grad()
            lossD.backward()
            optimD.step()

            # --- Generator 学習 ---
            z = make_noise(ranks)
            fake_data = netG(z)
            pred = netD(fake_data, ranks)
            lossG = criterion(pred, real_labels)

            optimG.zero_grad()
            lossG.backward()
            optimG.step()

        print(f"Epoch [{epoch}/{n_epochs}] | LossD: {lossD.item():.4f} | LossG: {lossG.item():.4f}")


def generate_samples(netG, rank, n_generate=10):
    labels = torch.tensor([rank]*n_generate, dtype=torch.long, device=device)
    z = make_noise(labels)
    fake_samples = netG(z)

    # binarization
    fake_samples[fake_samples < 0.5] = 0
    fake_samples[fake_samples >= 0.5] = 1

    # 0-1 -> atomic number
    fake_samples[fake_samples == 0] = ATOMIC_NUMBERS["Ni"]
    fake_samples[fake_samples == 1] = ATOMIC_NUMBERS["Pd"]

    return fake_samples


if __name__ == "__main__":
    #
    # Step 1. Prepare the dataset by generating the surface structures and calculating the activation energy.
    #
    num_samples = 10
    e_rxns = []
    atomic_numbers = []

    for i in range(num_samples):
        surf = fcc111("Au", size=(3, 3, 4), vacuum=10.0)  # element is dummy
        surf.pbc = True

        possible_elements = ["Ni", "Pd"]
        symbols = np.random.choice(possible_elements, len(surf), p=[0.9, 0.1])  # p: possibility of each element
        surf.set_chemical_symbols(symbols)

        e_rxn = get_reaction_energy(surface=surf)  # function to be implemented
        atomic_numbers.append(surf.get_atomic_numbers())  # get element information for surface

        print(f"Reaction energy of sample {i + 1}: {np.random.normal(0, 1)}")

        # add atomic numbers and reaction energy to the list
        atomic_numbers.append(surf.get_atomic_numbers())
        e_rxns.append(e_rxn)

    # Approximate the activation energies by linear relationship (alpha*reaction_energy + beta).
    # Here alpha and beta are parameters.
    alpha = 1.0
    beta = 1.6

    e_rxns = np.array(e_rxns)
    e_acts = alpha * e_rxns + beta

    print(f"Activation energies (in eV): {e_acts}")

    # Step 2. Make the descriptor-target pair, by setting atomic numbers as descriptors and activation energy as target.
    samples = make_samples(atomic_numbers, e_acts)

    # Step 3. Assign the smallest activation energy surfaces as "rank=0".
    samples = assign_rank(samples)

    # Step 4. Put above dataset and train the CGAN.
    batch_size = 10
    dataset = AtomicDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    nz = 100
    noise_std = 0.7
    sample_x, sample_energy, sample_rank = next(iter(dataloader))
    n_classes = len(torch.unique(torch.tensor([sample["rank"] for sample in samples])))
    image_size = len(sample_x[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eye = torch.eye(n_classes, device=device)

    print(f"Sample atomic numbers: {sample_x}")
    print(f"Sample activation energy: {sample_energy}")
    print(f"Sample rank: {sample_rank}")
    print(f"Number of classes: {n_classes}")
    print(f"Image size: {image_size}")

    netD = Discriminator().to(device)
    netG = Generator().to(device)
    optimD = optim.Adam(netD.parameters(), lr=0.0002)
    optimG = optim.Adam(netG.parameters(), lr=0.0002)
    n_epochs = 30
    train(netD, netG, optimD, optimG, n_epochs)

    # Step 5. Generate fake-samples for "rank=1" surfaces, to generate the surfaces with lower activation energy.
    rank_to_generate = 0
    n_generate = 4
    generated = generate_samples(netG, rank_to_generate, n_generate=n_generate)
    print(f"Generated samples (rank={rank_to_generate}): {generated}")
