import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

class MLP(nn.Module):

    def __init__(self, input_feat=64, output_feat=3):
        super(MLP, self).__init__()
        # input image features and 1 depth feature
        self.layer1 = nn.Conv1d(input_feat + 1, 512, 1)
        self.layer2 = nn.Conv1d(512, 256, 1)
        self.layer3 = nn.Conv1d(256, 128, 1)
        self.layer4 = nn.Conv1d(128 + 1, 128, 1)
        self.layer5 = nn.Conv1d(128, 128, 1)
        self.layer6 = nn.Conv1d(128, output_feat, 1)
        # self.layer5 = nn.Linear(256, 3)

        self.apply(init_weights)

    def forward(self, img_feat, z):
        '''

        :param img_feat: [B, C_in, N], N = num of sample points
        :param z: [B, 1, N]
        :return: [B, C_out, N], N predictions
        '''
        y = F.relu(self.layer1(torch.cat([img_feat, z], dim=1)))
        y = F.relu(self.layer2(y))
        y = F.relu(self.layer3(y))
        y = F.relu(self.layer4(torch.cat([y, z], dim=1)))
        y = F.relu(self.layer5(y))
        y = F.normalize(self.layer6(y), p=2, dim=1)

        return y


if __name__ == '__main__':
    mlp = MLP()
    sample = torch.randn((3, 64, 1000))
    z = torch.randn((3, 1, 1000))
    out = mlp(sample, z)
    print(out.size())

