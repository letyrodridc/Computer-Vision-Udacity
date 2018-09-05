import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        embeddings = self.word_embeddings(captions[:,:-1])
        embeddings = torch.cat([features.unsqueeze(1), embeddings.float()], 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    
    def sample(self, features, states=None, max_len=20):
        res =[]
        prev_state = None
        
        for i in range(max_len):
            hiddens, state = self.lstm(features, prev_state)
            outputs = self.linear(hiddens)
            _, predicted = torch.max(outputs,2)
            
           
            features = self.word_embeddings(predicted)
            
            prev_state = state
            
            res.append(predicted)
        
        res = torch.cat(res, 1)

        res = res.cpu().data.numpy()
     
  
        return res.tolist()[0]