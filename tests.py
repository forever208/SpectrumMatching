import torch
import unittest

from modules import LDMConfig, VAE, VQVAE, UNet2DModel

def generated_random_embedding(batch_size, embed_dim):
    return torch.randn(batch_size, embed_dim)

def generate_random_text_embeddings(batch_size, seq_len, embed_dim):
    attention_mask = torch.randint(low=0, high=2, size=(batch_size, seq_len)).bool()
    text_embeddings = torch.rand(batch_size, seq_len, embed_dim)
    return text_embeddings, attention_mask

def generate_random_images(batch, channels, height, width):
    return torch.rand(batch, channels, height, width)

class TestCases(unittest.TestCase):

    def test_unconditional_unet(self):

        config = LDMConfig(text_conditioning=False,
                           class_conditioning=False)
        
        model = UNet2DModel(config)

        compressed_image_size = config.img_size // 2**(len(config.vae_channels_per_block)-1)
        latent_shape = (2, config.latent_channels, compressed_image_size, compressed_image_size)

        images = generate_random_images(*latent_shape)
        time_embedding = generated_random_embedding(2, config.time_embed_proj_dim)

        out = model.forward(images, time_embedding)

        self.assertEqual(tuple(out.shape), latent_shape)

    def test_text_conditional_unet(self):

        config = LDMConfig(text_conditioning=True,
                           class_conditioning=False)
        
        model = UNet2DModel(config)

        compressed_image_size = config.img_size // 2**(len(config.vae_channels_per_block)-1)
        latent_shape = (2, config.latent_channels, compressed_image_size, compressed_image_size)

        images = generate_random_images(*latent_shape)
        time_embedding = generated_random_embedding(2, config.time_embed_proj_dim)
        text_embedding, text_attn_mask = generate_random_text_embeddings(2, 64, config.text_embed_dim)

        out = model.forward(images, 
                            time_embedding, 
                            text_conditioning=text_embedding, 
                            text_attention_mask=text_attn_mask)

        self.assertEqual(tuple(out.shape), latent_shape)

    def test_class_conditional_unet(self):

        config = LDMConfig(text_conditioning=False,
                           class_conditioning=True)
        
        model = UNet2DModel(config)

        compressed_image_size = config.img_size // 2**(len(config.vae_channels_per_block)-1)
        latent_shape = (2, config.latent_channels, compressed_image_size, compressed_image_size)

        images = generate_random_images(*latent_shape)
        time_embedding = generated_random_embedding(2, config.time_embed_proj_dim)
        class_embeddings = generated_random_embedding(2, config.class_embed_dim)

        out = model.forward(images, 
                            time_embedding,
                            class_conditioning=class_embeddings)

        self.assertEqual(tuple(out.shape), latent_shape)

    def test_text_and_class_conditional(self):
        
        config = LDMConfig(text_conditioning=True,
                           class_conditioning=True)
        
        model = UNet2DModel(config)

        compressed_image_size = config.img_size // 2**(len(config.vae_channels_per_block)-1)
        latent_shape = (2, config.latent_channels, compressed_image_size, compressed_image_size)

        images = generate_random_images(*latent_shape)
        time_embedding = generated_random_embedding(2, config.time_embed_proj_dim)
        class_embeddings = generated_random_embedding(2, config.class_embed_dim)
        text_embedding, text_attn_mask = generate_random_text_embeddings(2, 64, config.text_embed_dim)

        out = model.forward(images, 
                            time_embedding, 
                            text_embedding, 
                            text_attn_mask, 
                            class_embeddings)

        self.assertEqual(tuple(out.shape), latent_shape)

    def test_vae(self):

        config = LDMConfig()
        
        model = VAE(config)

        image_shape = (2, config.in_channels, config.img_size, config.img_size)
        images = generate_random_images(*image_shape)

        out = model(images)["reconstruction"]

        self.assertEqual(tuple(out.shape), image_shape)

    def test_vqvae(self):

        config = LDMConfig(quantize=True)
        
        model = VQVAE(config)

        image_shape = (2, config.in_channels, config.img_size, config.img_size)
        images = generate_random_images(*image_shape)

        out = model(images)["reconstruction"]

        self.assertEqual(tuple(out.shape), image_shape)
        
if __name__ == "__main__":
    unittest.main()