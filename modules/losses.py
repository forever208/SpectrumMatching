import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from .mylpips import LPIPS as mylpips
from .discriminator import PatchGAN, init_weights

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
    
    def forward(self, real, fake):
        loss_real = torch.mean(F.relu(1.0 - real))
        loss_fake = torch.mean(F.relu(1.0 + fake))
        return 0.5 * (loss_real + loss_fake)

class VanillaLoss(nn.Module):
    def __init__(self):
        super(VanillaLoss, self).__init__()

    def forward(self, real, fake):
        return 0.5 * (torch.mean(F.softplus(-real)) +
                      torch.mean(F.softplus(fake)))

class LpipsDiscriminatorLoss(nn.Module):
    def __init__(self, 
                 use_disc=True,
                 disc_in_channels=3, 
                 disc_start_dim=64, 
                 disc_depth=3, 
                 disc_kernel_size=3,
                 disc_leaky_relu_slope=0.2,
                 disc_loss="hinge",
                 disc_weight=1.0,
                 use_lpips=True, 
                 use_lpips_package=False, 
                 path_to_lpips_checkpoint="lpips_vgg.pt",
                 lpips_weight=1.0,
                 reconstruction_loss="l1"):
        
        super(LpipsDiscriminatorLoss, self).__init__()

        self.disc_weight = disc_weight
        self.lpips_weight = lpips_weight
        self.use_disc = use_disc
        self.use_lpips = use_lpips
        
        ### Load LPIPS Model ###
        if use_lpips:
            if use_lpips_package:
                self.lpips_model = lpips.LPIPS(net="vgg").eval()
            else:
                self.lpips_model = mylpips()
                self.lpips_model.load_checkpoint(path_to_lpips_checkpoint)
        
        ### Load Discriminator ###
        if use_disc:
            self.discriminator = PatchGAN(input_channels=disc_in_channels,
                                          start_dim=disc_start_dim, 
                                          depth=disc_depth, 
                                          kernel_size=disc_kernel_size, 
                                          leaky_relu_slope=disc_leaky_relu_slope).apply(init_weights)

        ### Discriminator Loss ###
        if disc_loss == "hinge":
            self.disc_loss = HingeLoss()
        elif disc_loss == "vanilla":
            self.disc_loss = VanillaLoss()
        else:
            raise ValueError("Select disc_loss between 'hinge' or 'vanilla'")
        
        ### Reconstruction Loss ###
        if reconstruction_loss == "l1":
            self.reconst_loss = nn.L1Loss(reduction="none")
        elif reconstruction_loss == "l2":
            self.reconst_loss = nn.MSELoss(reduction="none")
        else:
            raise ValueError("Select reconstruction_loss btwn 'l1' or 'l2'")

    def forward_discriminator(self, inputs):
        return self.discriminator(inputs)
    
    @torch.no_grad()
    def forward_lpips(self, 
                      inputs, 
                      reconstructions):
        
        output = self.lpips_model(inputs, reconstructions)
        
        return output
    
    def forward_perceptual_loss(self, 
                                images, 
                                reconstruction,
                                img_average=False):
        
        ### Compute Reconstruction Loss ###
        reconstruction_loss = self.reconst_loss(images, reconstruction)

        ### Compute Perceptual Loss ###
        lpips_loss = torch.zeros(size=(), device=images.device)
        if self.use_lpips:
            lpips_loss = self.forward_lpips(reconstruction, images)

        ### Put Together Reconstruction and Perceptual Losses ###
        perceptual_loss = reconstruction_loss + self.lpips_weight * lpips_loss
        ### Sum Together the Loss by Pixel and Average ###
        if img_average:
            perceptual_loss = perceptual_loss.sum() / perceptual_loss.shape[0]
            reconstruction_loss = reconstruction_loss.sum() / reconstruction_loss.shape[0]
            lpips_loss = lpips_loss.sum() / lpips_loss.shape[0]

        ### Or avereage all the Pixel Errors Together ###
        else:
            perceptual_loss = perceptual_loss.mean()
            reconstruction = reconstruction.mean()
            lpips_loss = lpips_loss.mean()

        return perceptual_loss, reconstruction_loss, lpips_loss

    def forward_generator_loss(self,
                               reconstruction, 
                               perceptual_loss,
                               last_layer):
        
        ### Real -> inf, Fake -> -inf (Hinge Loss) ###
        ### To fool our discriminator, we want to maximize the logits (towards inf) ###
        ### We typically minimize, so we multiply by -1
        
        logits_fake = self.discriminator(reconstruction)
        generator_loss = -torch.mean(logits_fake)

        ### Loss Balancing Between Generator and Discriminator ###
        adaptive_weight = self.calculate_adaptive_weights(perceptual_loss, 
                                                          generator_loss, 
                                                          last_layer)
        
        return generator_loss, adaptive_weight


    def calculate_adaptive_weights(self, 
                                   perceptual_loss, 
                                   generator_loss, 
                                   last_layer):
        
        perceptual_grad = torch.autograd.grad(perceptual_loss, last_layer, retain_graph=True)[0]
        generator_grad = torch.autograd.grad(generator_loss, last_layer, retain_graph=True)[0]

        adaptive_weight = torch.norm(perceptual_grad) / (torch.norm(generator_grad) + 1e-4)
        adaptive_weight = torch.clamp(adaptive_weight, 0.0, 1e4).detach()

        return adaptive_weight
    
    def forward_discriminator_loss(self, images, reconstruction):

        ### Pass Real/Fake images into discriminator ###
        ### detach so no gradients to the VAE ###
        logits_real = self.discriminator(images.detach())
        logits_fake = self.discriminator(reconstruction.detach())

        ### Compute Discriminator Loss ###
        loss = self.disc_loss(logits_real, logits_fake)
        
        return loss, logits_real.mean(), logits_fake.mean()
    

# class HingeLoss(nn.Module):
#     def __init__(self):
#         super(HingeLoss, self).__init__()
    
#     def forward(self, real, fake):
#         loss_real = torch.mean(F.relu(1.0 - real))
#         loss_fake = torch.mean(F.relu(1.0 + fake))
#         return 0.5 * (loss_real + loss_fake)

# class GANLoss(nn.Module):

#     def __init__(self):
#         super(GANLoss, self).__init__()

#         self.loss_func = nn.BCEWithLogitsLoss()
    
#     def forward(self, x, labels):
        
#         return self.loss_func(x, labels)


# class LpipsDiscriminatorLoss(nn.Module):
#     def __init__(self, 
#                  use_disc=True,
#                  disc_in_channels=3, 
#                  disc_start_dim=64, 
#                  disc_depth=3, 
#                  disc_kernel_size=3,
#                  disc_leaky_relu_slope=0.2,
#                  disc_label_smoothing=0.05,
#                  disc_loss=None,
#                  use_lpips=True, 
#                  use_lpips_package=False, 
#                  path_to_lpips_checkpoint="lpips_vgg.pt",
#                  lpips_weight=1.0,
#                  reconstruction_loss="l1"):
        
#         super(LpipsDiscriminatorLoss, self).__init__()

#         self.lpips_weight = lpips_weight
#         self.use_lpips = use_lpips
#         self.label_smoothing = disc_label_smoothing
        
#         ### Load LPIPS Model ###
#         if use_lpips:
#             if use_lpips_package:
#                 self.lpips_model = lpips.LPIPS(net="vgg").eval()
#             else:
#                 self.lpips_model = mylpips()
#                 self.lpips_model.load_checkpoint(path_to_lpips_checkpoint)
        
#         ### Load Discriminator ###
#         if use_disc:
#             self.discriminator = PatchGAN(input_channels=disc_in_channels,
#                                           start_dim=disc_start_dim, 
#                                           depth=disc_depth, 
#                                           kernel_size=disc_kernel_size, 
#                                           leaky_relu_slope=disc_leaky_relu_slope).apply(init_weights)

#         ### Discriminator Loss ###
#         self.disc_loss = GANLoss()
        
#         ### Reconstruction Loss ###
#         if reconstruction_loss == "l1":
#             self.reconst_loss = nn.L1Loss(reduction="none")
#         elif reconstruction_loss == "l2":
#             self.reconst_loss = nn.MSELoss(reduction="none")
#         else:
#             raise ValueError("Select reconstruction_loss btwn 'l1' or 'l2'")

#     def forward_discriminator(self, inputs):
#         return self.discriminator(inputs)
    
#     @torch.no_grad()
#     def forward_lpips(self, 
#                       inputs, 
#                       reconstructions):
        
#         output = self.lpips_model(inputs, reconstructions)
        
#         return output
    
#     def forward_perceptual_loss(self, 
#                                 images, 
#                                 reconstruction,
#                                 img_average=True):
        
#         ### Compute Reconstruction Loss ###
#         reconstruction_loss = self.reconst_loss(images, reconstruction)

#         ### Compute Perceptual Loss ###
#         lpips_loss = torch.zeros(size=(), device=images.device)
#         if self.use_lpips:
#             lpips_loss = self.forward_lpips(reconstruction, images)

#         ### Put Together Reconstruction and Perceptual Losses ###
#         perceptual_loss = reconstruction_loss + self.lpips_weight * lpips_loss

#         ### Sum Together the Loss by Pixel and Average ###
#         if img_average:
#             perceptual_loss = perceptual_loss.sum() / perceptual_loss.shape[0]
#             reconstruction_loss = reconstruction_loss.sum() / reconstruction_loss.shape[0]
#             lpips_loss = lpips_loss.sum() / lpips_loss.shape[0]

#         ### Or average all the Pixel Errors Together ###
#         else:
#             perceptual_loss = perceptual_loss.mean()
#             reconstruction = reconstruction.mean()
#             lpips_loss = lpips_loss.mean()

#         return perceptual_loss, reconstruction_loss, lpips_loss

#     def forward_generator_loss(self,
#                                reconstruction, 
#                                perceptual_loss,
#                                last_layer):
        
#         ### Compute Logits on Discriminator for Our Generated Images ###
#         logits_fake = self.discriminator(reconstruction)

#         ### We want our model to predict these are REAL, so use 1 as the label ###
#         fake_true_labels = torch.ones_like(logits_fake, device=logits_fake.device)

#         ### Compute BCE Loss ###
#         generator_loss = self.disc_loss(logits_fake, fake_true_labels)

#         ### Loss Balancing Between Generator and Discriminator ###
#         adaptive_weight = self.calculate_adaptive_weights(perceptual_loss, 
#                                                           generator_loss, 
#                                                           last_layer)
        
#         return generator_loss, adaptive_weight

#     def calculate_adaptive_weights(self, 
#                                    perceptual_loss, 
#                                    generator_loss, 
#                                    last_layer):
        
#         perceptual_grad = torch.autograd.grad(perceptual_loss, last_layer, retain_graph=True)[0]
#         generator_grad = torch.autograd.grad(generator_loss, last_layer, retain_graph=True)[0]

#         adaptive_weight = torch.norm(perceptual_grad) / (torch.norm(generator_grad) + 1e-4)
#         adaptive_weight = torch.clamp(adaptive_weight, 0.0, 1e4).detach()

#         return adaptive_weight
    
#     def forward_discriminator_loss(self, images, reconstruction):

#         ### Pass Real/Fake images into discriminator ###
#         ### detach so no gradients to the VAE ###
#         logits_real = self.discriminator(images.detach())
#         logits_fake = self.discriminator(reconstruction.detach())

#         ### True/Fake Labels ###
#         true_labels = torch.ones_like(logits_real, device=logits_real.device)
#         fake_labels = torch.zeros_like(logits_fake, device=logits_fake.device)

#         ### Compute Real/Fake Loss ###
#         real_loss = self.disc_loss(logits_real, true_labels) - self.label_smoothing
#         fake_loss = self.disc_loss(logits_fake, fake_labels) + self.label_smoothing

#         ### Compute Discriminator Loss ###
#         loss = (real_loss + fake_loss) / 2
        
#         return loss, logits_real.mean(), logits_fake.mean()