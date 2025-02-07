import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ase.build import fcc111

# --- Step 1. Prepare the dataset ---
num_samples = 10
e_ads = []
atomic_numbers = []

def get_adsorption_energy(surface):
    return np.random.normal(0, 1)

for i in range(num_samples):
    surf = fcc111('Pt', size=(2,2,3), vacuum=10.0)
    e_ads.append(get_adsorption_energy(surf))
    atomic_numbers.append(surf.get_atomic_numbers())

# --- Step 2. Make descriptor-target pairs ---
def make_samples(atomic_numbers, e_ads):
    samples = []
    for atom_num, e_ad in zip(atomic_numbers, e_ads):
        samples.append({
            'atomic_numbers': atom_num,
            'adsorption_energy': e_ad
        })
    return samples

samples = make_samples(atomic_numbers, e_ads)

# --- Step 3. Assign ranks starting from 0 ---
def assign_rank(samples):
    samples = sorted(samples, key=lambda x: x['adsorption_energy'])
    for i, sample in enumerate(samples):
        sample['rank'] = i
    return samples

samples = assign_rank(samples)

# --- Dataset and DataLoader ---
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

# --- Generator and Discriminator Models ---
nz = 100  # Noise vector size
noise_std = 0.7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_classes = len(torch.unique(torch.tensor([s['rank'] for s in samples])))

eye = torch.eye(n_classes, device=device)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nz + n_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 10),  # 3D coordinates for 10 atoms
            nn.Sigmoid()  # Position values between 0 and 1
        )

    def forward(self, z, labels):
        input = torch.cat([z, eye[labels]], dim=1)
        output = self.net(input)
        
        atomic_numbers = output[:, :10]  # Atomic numbers (10 atoms)
        positions = output[:, 10:].view(-1, 10, 3)  # Atomic positions (10 atoms, 3D coordinates)
        
        return atomic_numbers, positions

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * 10 + n_classes, 512),  # 10 atoms' positions and labels
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, atomic_numbers, positions, labels):
        input_data = torch.cat([positions.view(positions.size(0), -1), eye[labels]], dim=1)
        return self.net(input_data)

# --- Optimizers and Loss ---
netD = Discriminator().to(device)
netG = Generator().to(device)

optimD = optim.Adam(netD.parameters(), lr=0.0002)
optimG = optim.Adam(netG.parameters(), lr=0.0002)
criterion = nn.BCELoss()

real_labels = torch.ones(batch_size, 1, device=device)
fake_labels = torch.zeros(batch_size, 1, device=device)

# --- Noise Generation ---
def make_noise(labels):
    labels_onehot = eye[labels]  # Convert labels to one-hot vectors
    labels_onehot = labels_onehot.repeat_interleave(nz // n_classes, dim=-1)
    z = torch.normal(0, noise_std, size=(len(labels), nz), device=device)
    z = z + labels_onehot  # Add label information to the noise
    return z

# --- Training Loop ---
def train(netD, netG, optimD, optimG, n_epochs):
    netD.train()
    netG.train()
    for epoch in range(1, n_epochs + 1):
        for atomic_numbers, _, ranks in dataloader:
            atomic_numbers = atomic_numbers.to(device)
            ranks = ranks.to(device)

            # --- Discriminator Training ---
            z = make_noise(ranks)
            fake_atomic_numbers, fake_positions = netG(z, ranks)
            pred_fake = netD(fake_atomic_numbers, fake_positions, ranks)
            pred_real = netD(atomic_numbers, fake_positions, ranks)  # Real data

            lossD_fake = criterion(pred_fake, fake_labels)
            lossD_real = criterion(pred_real, real_labels)
            lossD = lossD_fake + lossD_real

            optimD.zero_grad()
            lossD.backward()
            optimD.step()

            # --- Generator Training ---
            z = make_noise(ranks)
            fake_atomic_numbers, fake_positions = netG(z, ranks)
            pred = netD(fake_atomic_numbers, fake_positions, ranks)
            lossG = criterion(pred, real_labels)

            optimG.zero_grad()
            lossG.backward()
            optimG.step()

        print(f"Epoch [{epoch}/{n_epochs}] | LossD: {lossD.item():.4f} | LossG: {lossG.item():.4f}")

# --- Train the models ---
n_epochs = 30
train(netD, netG, optimD, optimG, n_epochs)

# --- Generate Samples ---
def generate_samples(netG, rank, n_samples=10):
    labels = torch.tensor([rank] * n_samples, dtype=torch.long, device=device)
    z = make_noise(labels)
    fake_atomic_numbers, fake_positions = netG(z, labels)
    return fake_atomic_numbers, fake_positions

rank_to_generate = 0
generated_atomic_numbers, generated_positions = generate_samples(netG, rank_to_generate, 5)

print("Generated Atomic Numbers:")
print(generated_atomic_numbers)
print("Generated Atomic Positions:")
print(generated_positions)
