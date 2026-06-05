import json
import pathlib
import torch # for tensor operations
import torch.nn as nn # for building neural networks
from torch.nn import functional as F # for activation functions and other functional operations


batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000 # how many training iterations?
eval_interval = 300 # how often to evaluate the model?
learning_rate = 1e-3 # learning rate for optimization
device = torch.device('mps') # device to run the model on (GPU if available, otherwise CPU)
eval_iters = 200 # number of iterations to evaluate the model
n_embed = 32 # size of the embedding vectors

torch.manual_seed(1337) # set the random seed for reproducibility

with open('../input.txt', 'r', encoding='utf-8') as f: # open the input text file
    text = f.read() # read the contents of the file into a string


## Embedding Functions
chars = sorted(list(set(text))) # get a sorted list of unique characters in the text
vocab_size = len(chars) # the size of the vocabulary (number of unique characters)
stoi = {ch:i for i,ch in enumerate(chars)} # create a mapping from characters to integers
itos = {i:ch for i, ch in enumerate(chars)} # create a mapping from integers to characters
encode = lambda s : [stoi[c] for c in s] # function to encode a string into a list of integers
decode = lambda l : ''.join([itos[i] for i in l]) # function to decode a list of integers back into a string


## create the train and test splits

data = torch.tensor(encode(text), dtype=torch.long) # encode the entire text into a tensor of integers
n = int(0.9*len(data)) # split the data into training and validation sets (90% for training, 10% for validation)
train_data = data[:n] # training data
val_data = data[n:] # validation data

## batch generation

def get_batch(split): # function to generate a batch of data
    data = train_data if split == 'train' else val_data # select the appropriate dataset
    ix = torch.randint(len(data) - block_size, (batch_size,)) # randomly select starting indices for the batch
    x = torch.stack([data[i:i+block_size] for i in ix]) # create input sequences of length block_size
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # create target sequences (input shifted by one character)
    x,y = x.to(device), y.to(device) # move the input and target tensors to the specified device
    return x, y # return the input and target batches


## evaluation function
## We compute the loss across multiple batches and average them to get a more accurate estimate of the loss on the training and validation sets. 
# We also set the model to evaluation mode during this process to ensure that any layers that behave differently during training (like dropout) are handled correctly. 
# After evaluating, we set the model back to training mode before returning the loss estimates.

@torch.no_grad() # context manager to disable gradient calculation (we don't need gradients for evaluation)
def estimate_loss(): # function to estimate the loss on the training and validation sets
    out = {} # dictionary to store the loss values
    model.eval() # set the model to evaluation mode (disables dropout, etc.)
    for split in ['train', 'val']: # evaluate on both training and validation sets
        losses = torch.zeros(eval_iters) # tensor to store loss values for each evaluation iteration
        for k in range(eval_iters): # loop over the number of evaluation iterations
            X, Y = get_batch(split) # get a batch of data
            logits, loss = model(X, Y) # compute the logits and loss from the model
            losses[k] = loss.item() # store the loss value
        out[split] = losses.mean() # compute the mean loss for this split and store it in the output dictionary
    model.train() # set the model back to training mode before returning
    return out # return the dictionary containing the estimated losses 

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        v = self.value(x) # (B,T,head_size)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,head_size) @ (B, head_size, T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        out = wei @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out

class BigramLanguageModel(nn.Module): # define the bigram language model class
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # # create an embedding layer that maps each token to a vector of size n_embed
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # create an embedding layer that maps each position in the input sequence to a vector of size n_embed
        ## in here we don't want to get logits directly from the embedding table, we want to pass it through a feed forward network to get the logits, so we need to define that as well
        self.sa_head = Head(n_embed) # create a single head of self attention (for simplicity, we will use only one head and one layer of self attention)
        self.lm_head = nn.Linear(n_embed, vocab_size) # create a linear layer that maps the embedding vectors to the vocabulary size (logits for each token)

    def forward(self, idx, targets=None):
        
        B, T = idx.shape # get the batch size and sequence length from the shape of the input indices
        tok_emb = self.token_embedding_table(idx) # get the token embeddings for the input indices
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # get the position embeddings for each position in the input sequence
        x = tok_emb + pos_emb # combine the token and position embeddings by element-wise addition
        x = self.sa_head(x) # apply the self attention head to the combined embeddings
        logits = self.lm_head(x) # pass the combined embeddings through the linear layer to get the logits
        
        if targets is None: # if no targets are provided, return the logits and None for loss
            loss = None
        else:
            B, T, C = logits.shape # get the batch size, sequence length, and number of classes from the shape of the logits
            logits = logits.view(B*T, C) # reshape the logits to be a 2D tensor (batch_size * sequence_length, vocab_size)
            targets = targets.view(B*T) # reshape the targets to be a 1D tensor (batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets) # compute the cross-entropy loss between the logits and targets
        return logits, loss # return the logits and loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens): # loop over the number of new tokens to generate
            idx_cond = idx[:, -block_size:] # get the last block_size tokens from the input indices to use as context for generation
            logits, loss = self(idx_cond) # get the logits for the current input indices
            logits = logits[:, -1, :] # focus on the last time step's logits (the most recent token)
            probs = F.softmax(logits, dim=-1) # convert logits to probabilities using softmax
            idx_next = torch.multinomial(probs, num_samples=1) # sample the next token index from the probability distribution
            idx = torch.cat((idx, idx_next), dim=1) # append the new token index to the input indices for the next iteration
        return idx # return the generated sequence of indices   


## vocab_size is a global variable so no need to pass it as an argument to the model, we can just use it directly in the model definition
model = BigramLanguageModel().to(device) # create an instance of the bigram language model and move it to the specified device
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # create an AdamW optimizer to update the model parameters during training

_losses_dir = pathlib.Path(__file__).parent.parent / 'losses'
_losses_dir.mkdir(exist_ok=True)
_loss_file = _losses_dir / f'{pathlib.Path(__file__).stem}.json'
_loss_history = []

for iter in range(max_iters): # loop over the number of training iterations
    if iter % eval_interval == 0: # if it's time to evaluate the model
        losses = estimate_loss() # estimate the loss on the training and validation sets
        _loss_history.append({'step': iter, 'train': losses['train'].item(), 'val': losses['val'].item()})
        _loss_file.write_text(json.dumps(_loss_history, indent=2))
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") # print the current step and the estimated losses
    
    xb, yb = get_batch('train') # get a batch of training data
    logits, loss = model(xb, yb) # compute the logits and loss from the model
    optimizer.zero_grad(set_to_none=True) # zero out the gradients before backpropagation
    loss.backward() # perform backpropagation to compute gradients
    optimizer.step() # update the model parameters using the optimizer


context = torch.zeros((1, 1), dtype=torch.long).to(device) # create a tensor to hold the context for generation (starting with a single token)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist())) # generate a sequence of new tokens from the model and decode it back into a string, then print the generated text
