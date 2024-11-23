import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

class BiologicalAttention(nn.Module):
    """
    Memory-efficient attention mechanism for DNA sequences using linear attention.
    """
    def __init__(self, channels, heads=4):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        
        self.heads = heads
        self.channels_per_head = channels // heads
        self.scale = self.channels_per_head ** -0.5
        
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Optional positional encoding
        # self.pos_encoding = nn.Parameter(torch.randn(1, channels, 1))
        
    def forward(self, x):
        batch_size, C, L = x.size()
        
        # Add positional encoding
        # x = x + self.pos_encoding
        
        # Generate Q, K, V projections
        q = self.query(x).view(batch_size, self.heads, self.channels_per_head, L) 
        k = self.key(x).view(batch_size, self.heads, self.channels_per_head, L)
        v = self.value(x).view(batch_size, self.heads, self.channels_per_head, L)
        
        # Apply scaling
        q = q * self.scale
        
        # Efficient attention using cumulative sum
        k_cumsum = k.sum(dim=-1, keepdim=True)
        context = torch.einsum('bhcl,bhcd->bhcl', v, k_cumsum)
        
        # Normalize
        normalizer = torch.einsum('bhcl,bhcl->bhl', q, k_cumsum).unsqueeze(2)
        normalizer = torch.clamp(normalizer, min=1e-6)
        
        out = context / normalizer
        
        # Reshape and apply residual connection
        out = out.view(batch_size, C, L)
        return x + self.gamma * out

class DNAConvBlock(nn.Module):
    """
    DNA-specific convolutional block with dilated convolutions 
    to capture different ranges of nucleotide patterns
    """
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, 
            channels, 
            kernel_size=3, 
            padding='same',
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, 
            channels, 
            kernel_size=3, 
            padding='same',
            dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        return x + residual

class ImprovedGenerator(nn.Module):
    def __init__(self, latent_dim, num_channels, seq_len, vocab_size=4):
        super().__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels
        
        # Initial projection from latent space
        self.linear = nn.Linear(latent_dim, seq_len * num_channels)
        
        # DNA-specific processing blocks - now just using conv blocks
        self.blocks = nn.ModuleList([
            DNAConvBlock(num_channels, dilation=2**i)
            for i in range(3)  # 3 blocks with increasing dilation
        ])
        
        # Final projection to vocabulary size (4 for DNA: A,T,C,G)
        self.final_conv = nn.Conv1d(num_channels, vocab_size, 1)
        
    def forward(self, z):
        # Transform latent vector
        x = self.linear(z)
        x = x.view(-1, self.num_channels, self.seq_len)
        
        # Process through DNA-specific blocks
        for conv_block in self.blocks:
            x = conv_block(x)
            
        # Generate nucleotide probabilities
        x = self.final_conv(x)
        return F.softmax(x, dim=1)

class ImprovedCritic(nn.Module):
    def __init__(self, num_channels, seq_len, vocab_size=4):
        super().__init__()
        self.seq_len = seq_len
        reduced_channels = num_channels // 2
        
        # Initial projection from one-hot encoded DNA
        self.conv1 = nn.Conv1d(vocab_size, reduced_channels, 1)
        
        # DNA-specific processing blocks (without attention)
        self.blocks = nn.ModuleList([
            DNAConvBlock(reduced_channels, dilation=2**i)
            for i in range(3)
        ])
        
        # Global features extraction
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final scoring
        self.fc = nn.Linear(reduced_channels, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        
        # Process through conv blocks only
        for conv_block in self.blocks:
            x = conv_block(x)
            
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

class WGAN(L.LightningModule):
    def __init__(
        self,
        seq_len,
        gen_lr=2e-4,
        critic_lr=1e-4,
        vocab_size=4,
        latent_dim=128,
        lambda_gp=10,
        critic_iterations=5,
        num_channels=128
    ):
        super().__init__()
        self.automatic_optimization = False
        self.gen_lr = gen_lr
        self.critic_lr = critic_lr
        self.critic_iterations = critic_iterations
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        self.generator = ImprovedGenerator(
            latent_dim=latent_dim,
            num_channels=num_channels,
            seq_len=seq_len,
            vocab_size=vocab_size
        )
        
        self.critic = ImprovedCritic(
            num_channels=num_channels,
            seq_len=seq_len,
            vocab_size=vocab_size
        )
        
    def forward(self, z):
        return self.generator(z)
    
    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(
            self.generator.parameters(), 
            lr=self.gen_lr, 
            betas=(0.5, 0.9)
        )
        opt_critic = torch.optim.Adam(
            self.critic.parameters(), 
            lr=self.critic_lr, 
            betas=(0.5, 0.9)
        )
        return opt_gen, opt_critic
    
    def gradient_penalty(self, real, fake):
        """
        Calculates the gradient penalty loss for WGAN GP
        """
        real = real.requires_grad_()
        fake = fake.requires_grad_()
        
        BATCH_SIZE, C, L = real.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1)).to(self.device)
        interp = real * alpha + fake * (1 - alpha)
        
        interp_score = self.critic(interp)
        
        gradient = torch.autograd.grad(
            inputs=interp,
            outputs=interp_score,
            grad_outputs=torch.ones_like(interp_score),
            create_graph=True,
            retain_graph=True
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return gradient_penalty
    
    def training_step(self, batch, batch_idx):
        real = batch
        cur_batch_size = real.shape[0]
        
        opt_g, opt_c = self.optimizers()
        
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(self.critic_iterations):
            noise = torch.randn(cur_batch_size, self.latent_dim).to("cuda")
            fake = self(noise)
            critic_real = self.critic(real).reshape(-1)
            critic_fake = self.critic(fake).reshape(-1)
            gp = self.gradient_penalty(real, fake)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.lambda_gp * gp
            
            # Log metrics
            self.log("real_score", torch.mean(critic_real), prog_bar=True)
            self.log("fake_score", torch.mean(critic_fake), prog_bar=True)
            self.log("d_loss", loss_critic, prog_bar=True)
            
            # Optimization step
            opt_c.zero_grad()
            self.manual_backward(loss_critic, retain_graph=True)
            opt_c.step()
        
        # Train Generator: min -E[critic(gen_fake)]
        gen_fake = self.critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        
        # Log generator loss
        self.log("g_loss", loss_gen, prog_bar=True)
        
        # Optimization step
        opt_g.zero_grad()
        self.manual_backward(loss_gen)
        opt_g.step()