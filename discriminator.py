import torch.nn as nn
import torch.nn.functional as F
import torch
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
                            # nn.InstanceNorm2d(cout)
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
        if activation == 'leakyrelu':
            self.act = nn.LeakyReLU()
        if activation == 'tanh':
            self.act = nn.Tanh()
    
    def forward(self, x):
        out = self.dense_layer(x)
        return self.act(out)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.image_discriminator = nn.Sequential(
        	Conv2d(3, 32, kernel_size=3, stride=4, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            
			Conv2d(32, 64, kernel_size=3, stride=4, padding=1),
			Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
			Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

			Conv2d(64, 128, kernel_size=3, stride=4, padding=1),
			Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
			Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

			Conv2d(128, 256, kernel_size=3, stride=4, padding=1),
			Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
			Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

			Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
			Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
			Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

			nn.Flatten(),

            # Dense(in_features=512, out_features=128, bias=True, activation='leakyrelu'),
            # Dense(in_features=128, out_features=32, bias=True, activation='leakyrelu'),
            # Dense(in_features=32, out_features=8, bias=True, activation='leakyrelu'),
            # Dense(in_features=8, out_features=1, bias=True, activation='sigmoid')
        )

        self.image_discriminator_final = nn.Sequential(

            Dense(in_features=896, out_features=512, bias=True, activation='leakyrelu'),
            Dense(in_features=512, out_features=128, bias=True, activation='leakyrelu'),
            Dense(in_features=128, out_features=32, bias=True, activation='leakyrelu'),
            Dense(in_features=32, out_features=8, bias=True, activation='leakyrelu'),
            Dense(in_features=8, out_features=1, bias=True, activation='sigmoid')

            )
    def forward(self, image, src_sent):
        generated_image = self.image_discriminator(image)
        concatenated_data = torch.cat((generated_image, src_sent.squeeze(2).squeeze(2)), 1)
        generated_image = self.image_discriminator_final(concatenated_data)
        return generated_image

discriminator = Discriminator().to(device)
generated_image = torch.tensor(np.random.random((32, 3, 256, 256)).astype(np.float32)).to(device)
sentence_embedding = torch.tensor(np.random.random((32, 384, 1, 1)).astype(np.float32)).to(device)

output = discriminator.forward(generated_image, sentence_embedding)
# from torchviz import make_dot
# make_dot(output, params=dict(list(discriminator.named_parameters()))).render("discriminator", format="png")


