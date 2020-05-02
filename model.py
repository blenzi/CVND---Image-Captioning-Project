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
        # batch normalization
        self.batchNorm = nn.BatchNorm1d(embed_size,momentum = 0.01)
        # Weights initialization
        self.embed.weight.data.normal_(0., 0.02)
        self.embed.bias.data.fill_(0)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.batchNorm(self.embed(features))
        return features    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # The LSTM takes embedded features as inputs, and outputs hidden states
        # with dimensionality hidden_size.
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # The linear layer that maps from hidden state space to word space
        self.hidden2word = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeds = self.word_embeddings(captions[:, :-1])
        inputs = torch.cat([features.unsqueeze_(1), embeds], dim=1)
        outputs, _ = self.lstm(inputs.cuda() if torch.cuda.is_available() else inputs)
        return self.hidden2word(outputs)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = [] # initialize outputs
        for i in range(max_len):
            # Predict next word
            output, states = self.lstm(inputs, states) # output in hidden space
            output = self.hidden2word(output.squeeze(1)) # output scores in word space
            _, index = torch.max(output, 1) # word with maximum score
            outputs.append(index.item()) # append the result
            # if predicted word is <end>, break the loop
            if index == 1:
                break
            # embed the last predicted word to be the new input of LSTM
            inputs = self.word_embeddings(index).unsqueeze(1)
        
        return outputs
