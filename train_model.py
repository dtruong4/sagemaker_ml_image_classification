import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse

import smdebug.pytorch as smd

def test(model, test_loader, criterion, device):
    # Set our model to evaluation mode
    model.eval()

    # Initalize our running counts to 0
    running_loss = 0
    running_corrects = 0
    running_total = 0
    
    for inputs, labels in test_loader:
        # Set to data and labels to be device compatible
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Make label predictions with the model for the inputs and calculate loss
        outputs = model(inputs)

        # Calculate the cost associated with the predictions
        loss = criterion(outputs, labels)

        # Select the single label with the highest prediction value
        outputs = outputs.argmax(dim = 1, keepdim = True)

        # Increment running values
        running_loss += loss.item() * inputs.size(0)
        running_corrects += outputs.eq(labels.view_as(outputs)).sum().item()
        running_total += len(inputs)
    
    # Calculate and print testing results
    total_loss = running_corrects / running_total
    total_acc = running_corrects / running_total
    print(f'\nTest Set - Average Loss: {total_loss}, Accuracy: {running_corrects}/{running_total} ({100.0 * total_acc}%)\n')

def validate(model, valid_loader, criterion, device, hook):
    # Set our model to evaluation mode
    model.eval()
    
    # Set our debug hook to evaluation mode
    hook.set_mode(smd.modes.EVAL)

    # Initalize our running counts to 0
    running_loss = 0
    running_corrects = 0
    running_total = 0
    
    for inputs, labels in valid_loader:
        # Set to data and labels to be device compatible
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Make label predictions with the model for the inputs and calculate loss
        outputs = model(inputs)

        # Calculate the cost associated with the predictions
        loss = criterion(outputs, labels)

        # Select the single label with the highest prediction value
        outputs = outputs.argmax(dim = 1, keepdim = True)

        # Increment running values
        running_loss += loss.item() * inputs.size(0)
        running_corrects += outputs.eq(labels.view_as(outputs)).sum().item()
        running_total += len(inputs)
    
    # Calculate and print validation results
    total_loss = running_corrects / running_total
    total_acc = running_corrects / running_total
    print(f'Validation Set - Average Loss: {total_loss}, Validation Accuracy: {running_corrects}/{running_total} ({100.0 * total_acc}%)\n')
    
def train(model, train_loader, valid_loader, criterion, optimizer, epochs, device, hook):
    # Set our model to train mode
    model.train()

    for e in range(epochs):
        # Set our debug hook to train mode at the start of each epoch
        hook.set_mode(smd.modes.TRAIN)
        
        # Reset all of our running counts at the start of each epoch
        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        # Iterate through batch data
        for inputs, labels in train_loader:
            # Set to data and labels to be device compatible
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Reset the gradients of the optimizer to zero
            optimizer.zero_grad()

            # Make label predictions with the model for the inputs and calculate loss
            outputs = model(inputs)

            # Calculate the cost associated with the predictions
            loss = criterion(outputs, labels)
            
            # Calculate gradients and apply them to optimizer
            loss.backward()
            optimizer.step()

            # Select the single label with the highest prediction value
            outputs = outputs.argmax(dim = 1, keepdim = True)

            # Increment running values
            running_loss += loss.item() * inputs.size(0)
            running_corrects += outputs.eq(labels.view_as(outputs)).sum().item()
            running_total += len(inputs)
        
        # Calculate and print epoch-specific values
        epoch_loss = running_loss / running_total
        epoch_acc = running_corrects / running_total
        print(f'Training Set - Epoch {e} - Average Loss: {epoch_loss}, Accuracy: {running_corrects}/{running_total} ({100.0 * epoch_acc}%)')
        
        validate(model, valid_loader, criterion, device, hook)
    return model
    
def net(num_classes = 133):
    # Set up a pretrained ResNet model
    model = models.resnet18(pretrained = True)
    
    # Freeze the convolutional layers of our model
    for param in model.parameters():
        param.requires_grad = False
    
    # Define inputs for our fully-connected network
    num_features = model.fc.in_features
    
    # Set up our fully connected network layer
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace = True),
        nn.Linear(128, num_classes)
    )

    return model

def create_data_loaders(train_data_path, valid_data_path, test_data_path, batch_size):
    # Set up some transforms for preprocessing the data
    training_transform = transforms.Compose([
        # Randomly augment to increase diversity within training set
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomResizedCrop([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    validation_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    testing_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    # Set up our datasets
    train_dataset = torchvision.datasets.ImageFolder(
        root = train_data_path, 
        transform = training_transform
    )
    valid_dataset = torchvision.datasets.ImageFolder(
        root = valid_data_path, 
        transform = validation_transform
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root = test_data_path, 
        transform = testing_transform
    )

    # Set up our data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = batch_size,
        shuffle = True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size = batch_size,
        shuffle = True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = batch_size,
        shuffle = True
    )

    return train_loader, valid_loader, test_loader

def main(args):
    # Set the device to a GPU if available; otherwise, use a CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize a model and set it to our device
    model = net(args.num_classes)
    model.to(device)

    # Create and register a debugger hook for the model
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    
    # Create a loss criterion and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), args.lr)

    # Register the debugger hook for the loss function
    hook.register_loss(loss_criterion)

    # Set up the data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        args.train,
        args.valid,
        args.test, 
        args.batch_size
    )

    # Train the model with the data sets
    train(model, train_loader, valid_loader, loss_criterion, optimizer, args.epochs, device, hook)
    
    # Test the accuracy of the model
    test(model, test_loader, loss_criterion, device)
    
    # Save the trained model
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, 'model.pth'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    # Parse hyperparameters for model
    parser.add_argument(
        '--num-classes',
        type = int,
        default = 133,
        metavar = 'N',
        help = 'Number of classes to label images (default: 133)',
    )
    parser.add_argument(
        '--batch-size',
        type = int,
        default = 256,
        metavar = 'N',
        help = 'Input batch size for training and testing (default: 256)',
    )
    parser.add_argument(
        '--lr', 
        type = float, 
        default = 0.015, 
        metavar = 'LR', 
        help = 'Learning rate for optimizer (default: 0.015)'
    )
    parser.add_argument(
        '--epochs',
        type = int,
        default = 11,
        metavar = 'N',
        help = 'Number of epochs to train (default: 11)',
    )

    # Parse directory path names
    parser.add_argument(
        '--train', 
        type = str, 
        default = os.environ['SM_CHANNEL_TRAIN']
    )
    parser.add_argument(
        '--valid', 
        type = str, 
        default = os.environ['SM_CHANNEL_VALID']
    )
    parser.add_argument(
        '--test', 
        type = str, 
        default = os.environ['SM_CHANNEL_TEST']
    )
    parser.add_argument(
        '--model-dir', 
        type = str, 
        default = os.environ['SM_MODEL_DIR']
    )
    
    args = parser.parse_args()
    
    main(args)
