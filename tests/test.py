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
import pandas as pd

ATOMIC_NUMBERS = {"Ni": 28, "Ru": 44, "Rh": 45, "Pd": 46, "Pt": 78, "Au": 79}
LATTICE_CONSTANTS = {"Ni": 3.52, "Ru": 2.71, "Rh": 3.80, "Pd": 3.89, "Pt": 3.92, "Au": 4.08}
VACUUM = 9.0
SURF_SIZE = (3, 3, 4)

possible_elements = ["Pt", "Rh"]
# possible_elements = ["Pt", "Pd"]

np.set_printoptions(precision=3, suppress=True)

batch_size = 10
noise_std = 1.0
n_epochs = 500
n_rank = 5
nz = 100
lr = 1.0e-3
dropoutrate = 0.4  # default: 0.5
negative_slope = 0.01  # default: 0.01
rank_to_generate = 0
n_generate = 10
num_samples = 60
ratio = [0.7, 0.3]

num_steps_dft = 60
num_iteration = 3
latticeconstant = LATTICE_CONSTANTS[possible_elements[0]]*ratio[0] + LATTICE_CONSTANTS[possible_elements[1]]*ratio[1]

method = "m3gnet"  # emt or m3gnet or chgnet
reaction_type = "N2dissociation"  # "N2dissociation" or "O2dissociation"

# Approximate the activation energies by linear relationship (alpha*reaction_energy + beta).
# Here alpha and beta are parameters, known as "Universal Scaling Relationship".
alpha = 0.87
beta = 1.34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, image_size, n_classes):
        super().__init__()
        self.image_size = image_size
        self.n_classes = n_classes
        self.net = nn.Sequential(

            nn.Linear(image_size + n_classes, 128),
            nn.BatchNorm1d(128),  # need
            nn.LeakyReLU(negative_slope=negative_slope),  # need
            nn.Dropout(p=dropoutrate),  # test

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),  # need
            nn.LeakyReLU(negative_slope=negative_slope),  # need
            nn.Dropout(p=dropoutrate),  # test

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels, n_classes):
        labels_onehot = torch.eye(n_classes, device=device)[labels]
        x = torch.cat([x.view(x.size(0), -1), labels_onehot], dim=1)
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, image_size, n_classes):
        super().__init__()
        self.image_size = image_size
        self.n_classes = n_classes

        self.net = nn.Sequential(
            nn.Linear(nz, 128),
            nn.BatchNorm1d(128),  # need
            nn.LeakyReLU(negative_slope=negative_slope),  # need
            nn.Dropout(p=dropoutrate),  # test

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),  # need
            nn.LeakyReLU(negative_slope=negative_slope),  # need
            nn.Dropout(p=dropoutrate),  # test

            nn.Linear(128, image_size),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)


def make_noise(labels, n_classes):
    # ラベル情報をノイズに加える
    labels_onehot = torch.eye(n_classes, device=device)[labels]
    labels_onehot = labels_onehot.repeat_interleave(nz // n_classes, dim=-1)
    z = torch.normal(0, noise_std, size=(len(labels), nz), device=device)
    z = z + labels_onehot                              # (batch_size, nz)
    return z


def train(netD, netG, optimD, optimG, n_epochs, dataloader, n_classes):
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
            z = make_noise(ranks, n_classes)
            fake_data = netG(z)
            pred_fake = netD(fake_data, ranks, n_classes=n_classes)       # 偽データ 判定
            pred_real = netD(atomic_numbers_scaled, ranks, n_classes=n_classes)  # 本物データ 判定

            lossD_fake = criterion(pred_fake, fake_labels)
            lossD_real = criterion(pred_real, real_labels)
            lossD = lossD_fake + lossD_real

            optimD.zero_grad()
            lossD.backward()
            optimD.step()

            # --- Generator 学習 ---
            z = make_noise(ranks, n_classes)
            fake_data = netG(z)
            pred = netD(fake_data, ranks, n_classes=n_classes)
            lossG = criterion(pred, real_labels)

            optimG.zero_grad()
            lossG.backward()
            optimG.step()

        if (epoch % 20) == 0:
            print(f"Epoch [{epoch:3d}/{n_epochs:3d}] | LossD: {lossD.item():.4f} | LossG: {lossG.item():.4f}")


def generate_samples(netG, rank, n_generate=10, n_classes=0):
    labels = torch.tensor([rank]*n_generate, dtype=torch.long, device=device)
    z = make_noise(labels, n_classes=n_classes)
    fake_samples = netG(z)

    # binarization
    fake_samples[fake_samples < 0.5] = 0
    fake_samples[fake_samples >= 0.5] = 1

    # 0-1 -> atomic number
    fake_samples[fake_samples == 0] = ATOMIC_NUMBERS[possible_elements[0]]
    fake_samples[fake_samples == 1] = ATOMIC_NUMBERS[possible_elements[1]]

    return fake_samples


def add_activation_energy_to_dataframe(df, iteration, e_acts, formula_list):
    df_tmp = pd.DataFrame(columns=["iteration", "activation_energy", "formula"])
    for i, (e_act, formula) in enumerate(zip(e_acts, formula_list)):
        new_row = pd.DataFrame({
            "iteration": [iteration],
            "activation_energy": [e_act],
            "formula": [formula]
        })
        df_tmp = pd.concat([df_tmp, new_row], ignore_index=True)

    df_tmp = df_tmp.sort_values(by="activation_energy", ascending=False)
    df = pd.concat([df, df_tmp], ignore_index=True)

    return df


def make_barplot(df):
    # Create the bar plot

    import plotly.express as px

    fig = px.bar(df,
                 x=df.index,
                 y="activation_energy",
                 color="iteration",
                 barmode="group",
                 hover_data=["formula"],
                 labels={
                     "index": "Sample Index",
                     "activation_energy": "Activation Energy (eV)",
                     "iteration": "Iteration"
                 },
                 title="Activation Energies by Iteration")

    # Update layout for better appearance
    fig.update_layout(
        plot_bgcolor="white",
        showlegend=True,
        legend_title_text="Iteration",
        xaxis_gridcolor="lightgray",
        yaxis_gridcolor="lightgray"
    )

    # Show the plot
    fig.update_traces(width=0.5)
    fig.show()

    return None


def train_and_generate(samples=None):
    # Step 4. Put above dataset and train the CGAN.
    dataset = AtomicDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    sample_x, sample_energy, sample_rank = next(iter(dataloader))
    n_classes = len(torch.unique(torch.tensor([sample["rank"] for sample in samples])))
    image_size = len(sample_x[0])

    netD = Discriminator(image_size=image_size, n_classes=n_classes).to(device)
    netG = Generator(image_size=image_size, n_classes=n_classes).to(device)
    optimD = optim.Adam(netD.parameters(), lr=lr)
    optimG = optim.Adam(netG.parameters(), lr=lr)
    train(netD, netG, optimD, optimG, n_epochs, dataloader=dataloader, n_classes=n_classes)

    # Step 5. Generate fake-samples for "rank=1" surfaces, to generate the surfaces with lower activation energy.
    generated = generate_samples(netG, rank_to_generate, n_generate=n_generate, n_classes=n_classes)
    generated = generated.detach().numpy()

    return generated


if __name__ == "__main__":
    #
    # Step 1. Prepare the dataset by generating the surface structures and calculating the activation energy.
    #
    datum = []

    e_rxns = []
    atomic_numbers = []
    formula_list = []

    for isurf in range(num_samples):
        surf = fcc111("Au", size=SURF_SIZE, a=latticeconstant, vacuum=VACUUM)  # element is dummy
        surf.pbc = True
        symbols = np.random.choice(possible_elements, len(surf), p=ratio)
        surf.set_chemical_symbols(symbols)
        e_rxn = get_reaction_energy(surface=surf, method=method, steps=num_steps_dft, reaction_type=reaction_type)
        atomic_numbers.append(surf.get_atomic_numbers())
        formula_list.append(surf.get_chemical_formula())
        e_rxns.append(e_rxn)

    e_acts = alpha * np.array(e_rxns) + beta

    df = pd.DataFrame(columns=["iteration", "activation_energy", "formula"])
    df = add_activation_energy_to_dataframe(df, iteration=0, e_acts=e_acts, formula_list=formula_list)

    samples = make_samples(atomic_numbers, e_acts)  # setting atomic numbers as descriptors and e_acts as target.
    samples = assign_rank(samples)  # Assign the smallest activation energy surfaces as "rank=0".
    generated = train_and_generate(samples=samples)  # train GAN and generate fake samples

    # ---  Next iteration
    for i in range(num_iteration):
        print(f"Iteration {i + 1}")

        e_rxns = []
        atomic_numbers = []
        formula_list = []
        for isurf in generated:
            surf = fcc111("Au", size=SURF_SIZE, a=latticeconstant, vacuum=VACUUM)  # element is dummy
            surf.pbc = True
            surf.set_chemical_symbols(isurf)
            e_rxn = get_reaction_energy(surface=surf, method=method, steps=num_steps_dft, reaction_type=reaction_type)
            atomic_numbers.append(surf.get_atomic_numbers())
            formula_list.append(surf.get_chemical_formula())
            e_rxns.append(e_rxn)

        e_acts = alpha * np.array(e_rxns) + beta

        df = add_activation_energy_to_dataframe(df, iteration=i+1, e_acts=e_acts, formula_list=formula_list)

        newsamples = make_samples(atomic_numbers, e_acts)  # setting atomic numbers as descriptors and e_acts as target.
        samples += newsamples
        samples = assign_rank(samples)  # Assign the smallest activation energy surfaces as "rank=0".

        generated = train_and_generate(samples=samples)

    make_barplot(df)
