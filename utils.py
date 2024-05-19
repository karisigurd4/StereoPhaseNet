import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    print(f'Model loaded from {path}')
    return model
