import torch.optim as optim

def get_adam_optimizer(model, learning_rate=0.001):
    return optim.Adam(model.parameters(), lr=learning_rate)




