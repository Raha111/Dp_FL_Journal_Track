import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import copy
from models import SimpleNN

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = transform
)

test_dataset = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = transform
)

num_clients = 5
client_datasets = random_split(
    train_dataset,
    [len(train_dataset) // num_clients] * num_clients,
    generator=torch.Generator().manual_seed(42)
)



client_loaders = []

for client_dataset in client_datasets:
    loader = DataLoader(
        client_dataset,
        batch_size = 32,
        shuffle=True
    )

    client_loaders.append(loader)

test_loader = DataLoader(
    test_dataset,
    batch_size = 32,
    shuffle=False
)


#for i, client_dataset in enumerate(client_datasets):
#   print(f"Client {i+1} dataset size: {len(client_dataset)}")

global_model = SimpleNN()

def train_client(model, dataloader, local_epochs=1, lr = 0.01, device='cpu'):
    model.train()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(local_epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model



def federated_average(global_model, client_models, client_sizes):
    total_samples = sum(client_sizes)
    global_state = global_model.state_dict()

    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key]) 
        
        for client_model, client_size in zip(client_models, client_sizes):
            weight = client_size / total_samples
            global_state[key] += client_model.state_dict()[key] * weight


    global_model.load_state_dict(global_state)
    return global_model


client_sizes = [
    len(client_dataset)
    for client_dataset in client_datasets
]

#evaluation function

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

#multi_round training

num_rounds = 5

for round in range(num_rounds):

    client_models = []
    for i, client_loader in enumerate(client_loaders):
        print(f"Training client {i+1} for round {round+1}")
        local_model = copy.deepcopy(global_model)
        trained_model = train_client(local_model, client_loader, local_epochs=2)
        client_models.append(trained_model)
    global_model = federated_average(global_model, client_models, client_sizes)

    test_accuracy = evaluate_model(global_model, test_loader)
    print(f"Round {round+1}, Test accuracy: {test_accuracy:.2f}%")