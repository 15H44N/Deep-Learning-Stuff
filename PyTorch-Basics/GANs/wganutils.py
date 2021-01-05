import torch as torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    # print(f"Real Shape {real.shape} \n",f"Fake Shape {fake.shape} \n",f"eps Shape {alpha.shape} \n")
    interpolated_images = real * alpha + fake * (1 - alpha)
    # critic scores
    mixed_scores = critic(interpolated_images)

    # gradient of mixed scores WRT interpolated images
    gradient = torch.autograd.grad(
        inputs= interpolated_images,
        outputs= mixed_scores,
        grad_outputs= torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True
    )[0] 
    
    gradient = gradient.view(gradient.shape[0], -1) # flatten
    gradient_norm = gradient.norm(2, dim=1) # we take L2 Norm
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty