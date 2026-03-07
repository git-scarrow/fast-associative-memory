import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_tasks=None):
        super(MLP, self).__init__()
        self.num_tasks = num_tasks
        
        # Original GEM paper uses 2 layers of 256 for MNIST
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        if num_tasks is None:
            self.linear = nn.Linear(hidden_size, num_classes)
        else:
            self.linear = nn.ModuleList([
                nn.Linear(hidden_size, num_classes) for _ in range(num_tasks)
            ])

    def forward(self, x, task_id=0):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        
        if self.num_tasks is None:
            return self.linear(x)
        return self.linear[task_id](x)


class SmallCNN(nn.Module):
    def __init__(self, num_classes, num_tasks=None):
        super(SmallCNN, self).__init__()
        self.num_tasks = num_tasks
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        # 32x32 -> 16x16 -> 8x8
        # 64 channels * 8 * 8 = 4096
        
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        if num_tasks is None:
            self.linear = nn.Linear(512, num_classes)
        else:
            self.linear = nn.ModuleList([
                nn.Linear(512, num_classes) for _ in range(num_tasks)
            ])

    def forward(self, x, task_id=0):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        if self.num_tasks is None:
            return self.linear(x)
        return self.linear[task_id](x)
