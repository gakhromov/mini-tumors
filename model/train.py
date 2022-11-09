from model.model import *
from tqdm import tqdm
from datetime import date
import random

# Keep track of the numbers of epochs executed so far

def train_step(device, model, cross_entropy_loss, learning_rate, optimizer, train_dataloader, training_writer, NUM_STEPS, batch_size):
    """
    Train model for 1 epoch.
    """
    model.train()
    total_loss, total_accuracy = 0., 0.
    for i, (image, label) in enumerate(tqdm(train_dataloader)):

        image, label = image.to(device), label.to(device) # put the data on the selected execution device
        optimizer.zero_grad()   # zero the parameter gradients
        output = model(image)  # forward pass
        # if i == temp:
        #     print("output", [np.argmax(img.detach().numpy()) for img in output])
        #     print("output_raw", output)
        #     print("label", label)
        loss = cross_entropy_loss(output, label)    # compute loss
        # loss.register_hook(lambda grad: print(grad))
        loss.backward() # backward pass
        for name, param in model.named_parameters():
            print(name, param.grad)
        optimizer.step()    # perform update

        NUM_STEPS += 1

        training_writer.add_scalar('loss', loss.item(), NUM_STEPS)
        total_loss += loss.item()

        train_accuracy = (torch.argmax(output, dim=1) == label).float().sum() / len(label) #get the accuracy for the batch
        training_writer.add_scalar('accuracy', train_accuracy, NUM_STEPS)
        total_accuracy += (torch.argmax(output, dim=1) == label).float().sum()
    
    total_loss /= len(train_dataloader)
    total_accuracy /= (len(train_dataloader)*batch_size)

    return total_loss, total_accuracy


def evaluate(device, model, cross_entropy_loss, learning_rate, optimizer, val_dataloader, batch_size):
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
    total_accuracy /= (len(val_dataloader)*batch_size)

    return total_loss, total_accuracy

def train(device, model, cross_entropy_loss, learning_rate, optimizer, scheduler, n_epochs, train_dataloader, test_dataloader, save_path, NUM_EPOCH, NUM_STEPS, training_writer, validation_writer, batch_size):
    """
    Train and evaluate model.
    """

    for epoch in range(n_epochs):
        
        # train model for one epoch
        train_loss, train_accuracy = train_step(device, model, cross_entropy_loss, learning_rate, optimizer, train_dataloader, training_writer, NUM_STEPS, batch_size)
        scheduler.step()

        for name, weight in model.named_parameters():
            # Attach a lot of summaries to training_writer for TensorBoard visualizations.
            training_writer.add_scalar(f'{name}.mean', torch.mean(weight), NUM_EPOCH)
            training_writer.add_scalar(f'{name}.std_dev', torch.std(weight), NUM_EPOCH)
            training_writer.add_scalar(f'{name}.max', torch.max(weight), NUM_EPOCH)
            training_writer.add_scalar(f'{name}.min', torch.min(weight), NUM_EPOCH)
            training_writer.add_histogram(f'{name}.weights', weight, NUM_EPOCH)
            training_writer.add_histogram(f'{name}.grad', weight.grad, NUM_EPOCH)
        
        # evaluate    
        val_loss, val_accuracy = evaluate(device, model, cross_entropy_loss, learning_rate, optimizer, test_dataloader, batch_size)
        # log summaries to validation_writer
        validation_writer.add_scalar('loss', val_loss, NUM_STEPS)
        validation_writer.add_scalar('accuracy', val_accuracy, NUM_STEPS)

        print(f"[Epoch {NUM_EPOCH}] - Training : accuracy = {train_accuracy}, loss = {train_loss}", end=" ")
        print(f"Validation : accuracy = {val_accuracy}, loss = {val_loss}")

        NUM_EPOCH += 1
    if save_path != "" : torch.save(model.state_dict(), save_path + "_" + str(date.today()) + "_" + str(NUM_STEPS) + "_" + str(NUM_EPOCH) + ".pt")
    else: torch.save(model.state_dict(), "./model_weights/model_" + str(date.today()) + "_" + str(NUM_STEPS) + "_" + str(NUM_EPOCH) + ".pt")

    training_writer.flush()
    validation_writer.flush()
