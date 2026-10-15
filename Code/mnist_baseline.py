import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
'''
print("MNIST dataset loaded successfully.")
print("Training samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Label:", label)

print("Minimum pixel value:",image.min())
print("Maximum pixel value:",image.max())
'''
train_loader = DataLoader(
    train_dataset,
    batch_size = 32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 32,
    shuffle=False
)

images, labels = next(iter(train_loader))
'''
print("Batch of images shape:", images.shape)
print("Batch of labels shape:", labels.shape)
print("First 10 labels:",labels[:10])

'''

model = SimpleNN()
print(model)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, label in train_loader:
        outputs = model(images)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        prediction = outputs.argmax(dim=1)
        correct += (prediction == label).sum().item()
        total += label.size(0)

    accuracy = 100 * correct / total
    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")


model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, label in test_loader:
        outputs = model(images)
        prediction = outputs.argmax(dim=1)
        correct += (prediction == label).sum().item()
        total += label.size(0)

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")