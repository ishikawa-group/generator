import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ase.build import fcc111

# --- Step 1. Prepare the dataset
num_samples = 10
e_ads = []
atomic_numbers = []

def get_adsorption_energy(surface):
    return np.random.normal(0, 1)

for i in range(num_samples):
    surf = fcc111('Pt', size=(2,2,3), vacuum=10.0)
    e_ads.append(get_adsorption_energy(surf))
    atomic_numbers.append(surf.get_atomic_numbers())

# --- Step 2. Make descriptor-target pairs
def make_samples(atomic_numbers, e_ads):
    samples = []
    for atom_num, e_ad in zip(atomic_numbers, e_ads):
        samples.append({
            'atomic_numbers': atom_num,
            'adsorption_energy': e_ad
        })
    return samples

samples = make_samples(atomic_numbers, e_ads)

# --- Step 3. Assign ranks starting from 0
def assign_rank(samples):
    samples = sorted(samples, key=lambda x: x['adsorption_energy'])
    for i, sample in enumerate(samples):
        sample['rank'] = i
    return samples

samples = assign_rank(samples)

class AtomicDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        atomic_numbers = torch.tensor(self.samples[idx]['atomic_numbers'], dtype=torch.float32)
        energy = torch.tensor(self.samples[idx]['adsorption_energy'], dtype=torch.float32)
        rank = torch.tensor(self.samples[idx]['rank'], dtype=torch.long)
        return atomic_numbers, energy, rank

batch_size = 2
dataset = AtomicDataset(samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

nz = 100
noise_std = 0.7

sample_x, sample_energy, sample_rank = next(iter(dataloader))
print(f"Sample atomic numbers: {sample_x}")
print(f"Sample adsorption energy: {sample_energy}")
print(f"Sample rank: {sample_rank}")

n_classes = len(torch.unique(torch.tensor([s['rank'] for s in samples])))
image_size = len(sample_x[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Number of classes: {n_classes}")
print(f"Image size: {image_size}")

eye = torch.eye(n_classes, device=device)

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

netD = Discriminator().to(device)
netG = Generator().to(device)

optimD = optim.Adam(netD.parameters(), lr=0.0002)
optimG = optim.Adam(netG.parameters(), lr=0.0002)
criterion = nn.BCELoss()

# ラベル情報をノイズに加える
def make_noise(labels):
    labels_onehot = eye[labels]                        # (batch_size, n_classes)
    labels_onehot = labels_onehot.repeat_interleave(nz // n_classes, dim=-1)
    z = torch.normal(0, noise_std, size=(len(labels), nz), device=device)
    z = z + labels_onehot                              # (batch_size, nz)
    return z

def make_false_labels(labels):
    diff = torch.randint(1, n_classes, size=labels.size(), device=device)
    fake_labels = (labels + diff) % n_classes
    return fake_labels

n_epochs = 30
real_labels = torch.ones(batch_size, 1, device=device)
fake_labels = torch.zeros(batch_size, 1, device=device)

def train(netD, netG, optimD, optimG, n_epochs):
    netD.train()
    netG.train()
    for epoch in range(1, n_epochs+1):
        for atomic_numbers, _, ranks in dataloader:
            atomic_numbers = atomic_numbers.to(device)
            ranks = ranks.to(device)

            # --- Discriminator 学習 ---
            z = make_noise(ranks)
            fake_data = netG(z)
            pred_fake = netD(fake_data, ranks)     # 偽データ 判定
            pred_real = netD(atomic_numbers, ranks) # 本物データ 判定

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

train(netD, netG, optimD, optimG, n_epochs)

def generate_samples(netG, rank, n_samples=10):
    labels = torch.tensor([rank]*n_samples, dtype=torch.long, device=device)
    z = make_noise(labels)
    fake_samples = netG(z)
    return fake_samples

rank_to_generate = 0
generated = generate_samples(netG, rank_to_generate, 10)
print(f"Generated samples (rank={rank_to_generate}): {generated}")