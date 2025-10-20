import torch
import torch.nn as nn

class CRNN_CTC(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        # Simple conv stack to reduce H to 1 and keep temporal width
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),  # H/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),  # H/4
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1)),  # H/8
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2,1)),  # H/16
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(inplace=True),  # try to reach H≈1
        )
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B,1,H,W)
        feats = self.cnn(x)             # (B,C,H',W')
        B, C, Hp, Wp = feats.shape
        feats = feats.mean(dim=2)       # avg over height -> (B,C,W')
        feats = feats.permute(2,0,1)    # (T=B? no) -> (T=W', B, C)
        seq, _ = self.rnn(feats)        # (T,B,2*hidden=512)
        logits = self.fc(seq)           # (T,B,num_classes)
        T = logits.size(0)
        return logits, T
