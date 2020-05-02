version = '2'
batch_size = 64           # batch size
vocab_threshold = 5       # minimum word count threshold
vocab_from_file = True    # if True, load existing vocab file
embed_size = 512           # dimensionality of image and word embeddings
hidden_size = 512          # number of features in hidden state of the RNN decoder
LSTM_layers = 2            # number of LSTM layers in RNN decoder
LSTM_dropout = 0.1         # dropout probability for each LSTM layer
num_epochs = 1             # number of training epochs
save_every = 1             # determines frequency of saving model weights
print_every = 100          # determines window for printing average loss
log_file = f'training_log_v{version}.txt'       # name of file with saved training loss and perplexity

# Optimizer
optimizer_params = dict() #dict(betas=(0.9, 0.99), weight_decay=1e-2, lr=3e-4)
useScheduler = False
