from model_basic import *
from tqdm import tqdm

# Keep track of the numbers of epochs executed so far

def train_step(device, model, cross_entropy_loss, learning_rate, optimizer, train_dataloader):
    """
    Train model for 1 epoch.
    """

    model.train()
    for i, (image, label) in enumerate(tqdm(train_dataloader)):
        image, label = image.to(device), label.to(device) # put the data on the selected execution device
        optimizer.zero_grad()   # zero the parameter gradients
        output = model(image)  # forward pass
        loss = cross_entropy_loss(output, label)    # compute loss
        loss.backward() # backward pass
        optimizer.step()    # perform update

        train_accuracy = (torch.argmax(output, dim=1) == label).float().sum() / len(label) #get the accuracy for the batch
    
    return loss, train_accuracy


def evaluate(device, model, cross_entropy_loss, learning_rate, optimizer, val_dataloader):
    """
    Evaluate model on validation data.
    """
    model.eval()
    total_loss, total_accuracy = 0., 0.
    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(val_dataloader)):
            image, label = image.to(device), label.to(device) # put the data on the selected execution device
            output = model(image)  # forward pass
            loss = cross_entropy_loss(output, label)    # compute loss
            total_loss += loss.item()
            total_accuracy += (torch.argmax(output, dim=1) == label).float().sum()
        
    total_loss /= len(val_dataloader)
    total_accuracy /= (len(val_dataloader)*64) #change *64

    return total_loss, total_accuracy

def train(device, model, cross_entropy_loss, learning_rate, optimizer, n_epochs, train_dataloader, test_dataloader, save_path, NUM_EPOCH):
    """
    Train and evaluate model.
    """
    # use the global NUM_STEPS, NUM_EPOCH variable

    for epoch in range(n_epochs):
        
        # train model for one epoch
        train_loss, train_accuracy = train_step(device, model, cross_entropy_loss, learning_rate, optimizer, train_dataloader)

        val_loss, val_accuracy = evaluate(device, model, cross_entropy_loss, learning_rate, optimizer, test_dataloader)

        print(f"[Epoch {NUM_EPOCH}] - Training : accuracy = {train_accuracy}, loss = {train_loss}", end=" ")
        print(f"Validation : accuracy = {val_accuracy}, loss = {val_loss}")

        NUM_EPOCH += 1
    if save_path != "" : torch.save(model.state_dict(), save_path)
    else: torch.save(model.state_dict(), "./model_weights/model_" + str(NUM_EPOCH) + ".pt")