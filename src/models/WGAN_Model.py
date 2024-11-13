"""
Implementation of WGAN with gradient penalty
"""
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


# Construct residual block
class ResBlock(nn.Module):
    def __init__(self, num_channels, res_rate=0.3):
        super(ResBlock, self).__init__()
        self.num_channels = num_channels
        self.res_rate = res_rate
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=5, stride=1, padding='same'),
            nn.ReLU()
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=5, stride=1, padding='same'),
            nn.ReLU()
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.uniform_(m.weight, 
                                 -torch.sqrt(torch.tensor(3.)) * torch.sqrt(4. / torch.tensor(5 * self.num_channels + 5 * self.num_channels)), 
                                 torch.sqrt(torch.tensor(3.)) * torch.sqrt(4. / torch.tensor(5 * self.num_channels + 5 * self.num_channels)))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = self.conv_block1(x)
        res = self.conv_block2(res)

        return x + self.res_rate * res
    
    
# Construct residual block for critic
class ResBlockCritic(nn.Module):
    def __init__(self, num_channels, res_rate=0.3):
        super(ResBlockCritic, self).__init__()
        self.num_channels = num_channels
        self.res_rate = res_rate
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(0.2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(0.2)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.uniform_(m.weight, 
                                 -torch.sqrt(torch.tensor(3.)) * torch.sqrt(4. / torch.tensor(5 * self.num_channels + 5 * self.num_channels)), 
                                 torch.sqrt(torch.tensor(3.)) * torch.sqrt(4. / torch.tensor(5 * self.num_channels + 5 * self.num_channels)))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = self.conv_block1(x)
        res = self.conv_block2(res)

        return x + self.res_rate * res


# Construct generator with res blocks
class Generator(nn.Module):
    def __init__(self, latent_dim, num_channels, seq_len, res_rate, vocab_size=4, res_layers=5):
        super().__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels
        
        # Linear layer to transform the latent vector
        self.linear = nn.Linear(latent_dim, seq_len * num_channels)
        
        self.res_blocks = nn.ModuleList([ResBlock(num_channels, res_rate) for _ in range(res_layers)])
        
        self.conv = nn.Conv1d(num_channels, vocab_size, kernel_size=1, stride=1, padding='same')
        
    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.num_channels, self.seq_len)
        
        for res_block in self.res_blocks:
            x = res_block(x)
            
        x = self.conv(x)
        
        return F.softmax(x, dim=1)
        
    
# Construct Critic(discriminator)
class Critic(nn.Module):
    def __init__(self, num_channels, seq_len, vocab_size, res_rate=0.3, res_layers=5):
        super(Critic, self).__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels

        # Initial convolution layer
        self.conv1 = nn.Conv1d(vocab_size, num_channels, kernel_size=1, stride=1, padding='same')

        # Residual blocks
        self.res_blocks = nn.ModuleList([ResBlockCritic(num_channels, res_rate) for _ in range(res_layers)])

        # Final linear layer for scoring
        self.fc = nn.Linear(seq_len * num_channels, 1)
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 
                                 -torch.sqrt(torch.tensor(3.)) * torch.sqrt(2. / torch.tensor(m.weight.size(0) + m.weight.size(1))), 
                                  torch.sqrt(torch.tensor(3.)) * torch.sqrt(2. / torch.tensor(m.weight.size(0) + m.weight.size(1))))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        # Flatten the output for the linear layer
        x = x.view(-1, self.seq_len * self.num_channels)

        score = self.fc(x)
        return score

    
class WGAN(L.LightningModule):
    def __init__(
        self,
        seq_len,
        gen_lr = 1e-4,
        critic_lr = 1e-4,
        vocab_size = 4,
        latent_dim = 100,
        lambda_gp = 10,
        critic_iterations = 5,
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
        
        # Initialize networks
        self.generator = Generator(
            latent_dim=latent_dim, 
            num_channels=100, 
            seq_len=seq_len, 
            res_rate=0.3, 
            vocab_size=vocab_size
        )
        
        self.critic = Critic(
            num_channels=100, 
            seq_len=seq_len, 
            vocab_size=vocab_size, 
            res_rate=0.3
        )
        
    def forward(self, z):
        return self.generator(z)
    
    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(0.5, 0.9))
        opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, betas=(0.5, 0.9))
        
        return opt_gen, opt_critic
    
    def gradient_penalty(self, real, fake):
        real = real.requires_grad_()
        fake = fake.requires_grad_()
        
        BATCH_SIZE, C, L = real.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1)).to("cuda")
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
    
    def training_step(self, batch):
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
            self.log("real_score", torch.mean(critic_real), prog_bar=True)
            self.log("fake_score", torch.mean(critic_fake), prog_bar=True)
            self.log("d_loss", loss_critic, prog_bar=True)
            opt_c.zero_grad()
            self.manual_backward(loss_critic, retain_graph=True)
            opt_c.step()
        
        # Train generator
        gen_fake = self.critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        self.log("g_loss", loss_gen, prog_bar=True)
        opt_g.zero_grad()
        self.manual_backward(loss_gen)
        opt_g.step()
    