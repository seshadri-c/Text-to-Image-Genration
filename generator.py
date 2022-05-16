import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchsummary import summary

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda)) 

device = torch.device("cuda" if use_cuda else "cpu")

class Conv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm1d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.LeakyReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.Tanh()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation = 'sigmoid',  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_layer = nn.Linear(in_features, out_features)
        
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
    
    def forward(self, x):
        out = self.dense_layer(x)
        return self.act(out)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()


        self.noise_encoder = nn.Sequential(

            Conv1d(100, 128, kernel_size=1, stride=3, padding=1),
            Conv1d(128, 128, kernel_size=1, stride=3, padding=1, residual=True),
            Conv1d(128, 128, kernel_size=1, stride=3, padding=1, residual=True),

            Conv1d(128, 256, kernel_size=1, stride=3, padding=1),
            Conv1d(256, 256, kernel_size=1, stride=3, padding=1, residual=True),
            Conv1d(256, 256, kernel_size=1, stride=3, padding=1, residual=True),

            Conv1d(256, 384, kernel_size=1, stride=3, padding=1),
            Conv1d(384, 384, kernel_size=1, stride=3, padding=1, residual=True),
            Conv1d(384, 384, kernel_size=1, stride=3, padding=1, residual=True),

        )


        self.image_generator = nn.Sequential(

            Conv2dTranspose(768, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            # Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            # Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2dTranspose(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            # Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2dTranspose(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            # Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2dTranspose(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            # Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2dTranspose(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            Conv2d(8, 8, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(8, 8, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(4, 3, kernel_size=3, stride=1, padding=1),
            Conv2d(3, 3, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(3, 3, kernel_size=3, stride=1, padding=1, residual=True),

            nn.Flatten(),

            Dense(12288, 12288, activation="tanh")

        )

    def forward(self, sentence_embedding):

        mu, sigma = 0, 1
        noise_vector = torch.tensor(np.random.normal(mu, sigma, (sentence_embedding.shape[0], 100,1)).astype(np.float32)).to(device)
        noise_embedding = self.noise_encoder(noise_vector).unsqueeze(3)
        final_embedding = torch.cat((noise_embedding, sentence_embedding), dim=1)
        image_generated = self.image_generator(final_embedding)
        return image_generated


generator = Generator().to(device)
sentence_embedding = torch.tensor(np.random.random((4, 384, 1, 1)).astype(np.float32)).to(device)
generated_image = generator.forward(sentence_embedding)
# print("Input Shape : {} Generated Image : {}".format(sentence_embedding.shape, generated_image.shape))
# print(summary(generator,(384,1,1)))
from torchviz import make_dot
make_dot(generated_image, params=dict(list(generator.named_parameters()))).render("generator", format="pdf")