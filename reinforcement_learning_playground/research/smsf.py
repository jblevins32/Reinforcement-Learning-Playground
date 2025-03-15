import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SMSF():
    def __init__(self):
        s=1

    def InputToSMSF(self, states):
        
        # Isolate robot positions from states specifically for my env with 2 agents
        x1 = states[0,0]
        y1 = states[0,1]
        x2 = states[1,0]
        y2 = states[1,1]

        SMSF = x1 + y1**2 + x2**3 + y2**4

        return SMSF

    def CompareSMSF(self, states1, states2):
         
        return states1 != states2
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.num_attack_types = 2
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.discriminator_model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,out_features=self.num_attack_types),
        ).to(self.device)
    
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.002)

        self.criteria = nn.CrossEntropyLoss()

    def Discriminate(self, SMSF1, SMSF2):

        model_in = torch.tensor(np.array([SMSF1, SMSF2])).to(self.device).to(dtype=torch.float)

        predicted_labels = self.discriminator_model(model_in.unsqueeze(0))
        predicted_labels = F.softmax(predicted_labels, dim=1)

        return predicted_labels

    def Loss(self, predicted_labels, true_labels):

        return self.criteria(predicted_labels, true_labels)
    
    def Update(self, loss):

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()






