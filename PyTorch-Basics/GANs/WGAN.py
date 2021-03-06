"""
Mostly everything is from DC GAN implementation, except training is changed
Also, HP is changed per the paper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.serialization import load
from torch.utils import data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter
from dcganmodels import Critic, Discriminator, Generator, initialize_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        )
    ]
)

print(f"Using Device {device}")
# dataset = datasets.MNIST(root="../../Datasets", train=True, transform=transforms, download=True)
dataset = datasets.ImageFolder("../../Datasets/CelebA", transform = transforms)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_critic=  optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train(),
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        
        ## Train Critic
        ## We want to train the Critic more, 
        ## max E[critic(real)] - E[critic(fake)]
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            
            # need to maximize inside exp, so minimize using opt the -ve of that
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) #
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            
            #clipping weight
            for p in critic.parameters(): 
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP) # _ means inplace operation
       
        ## Train Generator
        ## min -E[critic(gen_fake)]
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
          
        if batch_idx % 100 == 0:
        # for tensorboard
            print(
                f"Epoch [{epoch}][{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \ "
                f"Loss D: {loss_critic: .4f}, Loss G: {loss_gen: .4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True)
                
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True)

                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Real Images", img_grid_real, global_step=step)
                step += 1
