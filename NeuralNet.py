import torch
import torch.nn as nn

landmark_dict = {"WRIST": 0,
                         "THUMB_CMC": 1,
                         "THUMB_MCP": 2,
                         "THUMB_IP": 3,
                         "THUMB_TIP": 4,
                         "INDEX_FINGER_MCP": 5,
                         "INDEX_FINGER_PIP": 6,
                         "INDEX_FINGER_DIP": 7,
                         "INDEX_FINGER_TIP": 8,
                         "MIDDLE_FINGER_MCP": 9,
                         "MIDDLE_FINGER_PIP": 10,
                         "MIDDLE_FINGER_DIP": 11,
                         "MIDDLE_FINGER_TIP": 12,
                         "RING_FINGER_MCP": 13,
                         "RING_FINGER_PIP": 14,
                         "RING_FINGER_DIP": 15,
                         "RING_FINGER_TIP": 16,
                         "PINKY_MCP": 17,
                         "PINKY_PIP": 18,
                         "PINKY_DIP": 19,
                         "PINKY_TIP": 20}

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21*3, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 12)
        )
        self.linear_dropout_stack = nn.Sequential(
            nn.Linear(21*3, 128),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.Dropout(),
            nn.Linear(32, 12)
        )
        self.cnn_stack = nn.Sequential(
            nn.Conv1d(3, 32, 2, stride=2),
            #nn.MaxPool1d(3),
            nn.Linear(10, 32),
            nn.Dropout(),
            nn.Flatten(start_dim=0), # Why the fuck do i need flatten in sequential module???
            nn.Linear(1024, 12)
        )

    def forward(self, x):
        #x = self.flatten(x)
        #logits = self.linear_relu_stack(x)
        #logits = self.linear_dropout_stack(x)
        logits = self.cnn_stack(torch.permute(x, (1,0)))
        return logits