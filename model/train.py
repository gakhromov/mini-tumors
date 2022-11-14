from model.model import *
from tqdm import tqdm
from datetime import date
import random
import os
import wandb

#wandb.init(entity="titty-twisters", project="my-test-project")
wandb.init(project="my-test-project")

# Keep track of the numbers of epochs executed so far

def train_step(device, model, cross_entropy_loss, learning_rate, optimizer, train_dataloader, NUM_STEPS, batch_size):
    """
    Train model for 1 epoch.
    """
    model.train()
    total_loss, total_accuracy = 0., 0.
    temp = random.randint(0,10)
    for i, (image, label) in enumerate(tqdm(train_dataloader)):
        for j in range(image.shape[0]):
            image[j] -= image[j].min(1, keepdim=True)[0]
            image[j] /= image[j].max(1, keepdim=True)[0]
        image, label = image.to(device), label.to(device) # put the data on the selected execution device
        optimizer.zero_grad()   # zero the parameter gradients
        output = model(image)  # forward pass
        # if i == temp:
        #     print("output", [np.argmax(img.detach().numpy()) for img in output])
        #     print("output_raw", output)
        #     print("label", label)
        loss = cross_entropy_loss(output, label)    # compute loss
        total_loss += loss.item()
        loss.backward() # backward pass
        # for name, param in model.named_parameters():
        #     print(name, param.grad)
        optimizer.step()    # perform update

        NUM_STEPS += 1

        train_accuracy = (torch.argmax(output, dim=1) == label).float().sum() / len(label) #get the accuracy for the batch
        total_accuracy += (torch.argmax(output, dim=1) == label).float().sum()

        wandb.log({"training":{"loss": loss.item(),
                    "training_accuracy": (torch.argmax(output, dim=1) == label).float().sum() / len(label),
                    "inputs": wandb.Image(image[0][0]),
                    "logits": wandb.Html(str(torch.argmax(output, dim=1))),
                    "label": wandb.Html(str(label))
        }}, step=NUM_STEPS)
    
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
            for j in range(image.shape[0]):
                image[j] -= image[j].min(1, keepdim=True)[0]
                image[j] /= image[j].max(1, keepdim=True)[0]
            image, label = image.to(device), label.to(device) # put the data on the selected execution device
            output = model(image)  # forward pass
            loss = cross_entropy_loss(output, label)    # compute loss
            total_loss += loss.item()
            total_accuracy += (torch.argmax(output, dim=1) == label).float().sum()
        
    total_loss /= len(val_dataloader)
    total_accuracy /= (len(val_dataloader)*batch_size)

    return total_loss, total_accuracy

def train(device, model, cross_entropy_loss, learning_rate, optimizer, scheduler, n_epochs, train_dataloader, test_dataloader, save_path, NUM_EPOCH, NUM_STEPS, batch_size):
    """
    Train and evaluate model.
    """

    for epoch in range(n_epochs):
        
        # train model for one epoch
        train_loss, train_accuracy = train_step(device, model, cross_entropy_loss, learning_rate, optimizer, train_dataloader, NUM_STEPS, batch_size)
        scheduler.step()

        # for name, weight in model.named_parameters():
        #     wandb.log({str(name) + ".mean": torch.mean(weight), 
        #                 str(name) + ".std_dev": torch.std(weight),
        #                 str(name) + ".max": torch.max(weight),
        #                 str(name) + ".min": torch.min(weight),
        #                 str(name) + ".weights": weight,
        #                 str(name) + ".grad": weight.grad,
        #                 "epochs": NUM_EPOCH
        #     })
        
        # evaluate    
        val_loss, val_accuracy = evaluate(device, model, cross_entropy_loss, learning_rate, optimizer, test_dataloader, batch_size)
        wandb.log({"validation":{'val_loss': val_loss,
                    'val_accuracy': val_accuracy
        }}, step=NUM_EPOCH)

        print(f"[Epoch {NUM_EPOCH}] - Training : accuracy = {train_accuracy}, loss = {train_loss}", end=" ")
        print(f"Validation : accuracy = {val_accuracy}, loss = {val_loss}")

        NUM_EPOCH += 1
    if save_path != "" : torch.save(model.state_dict(), save_path + "_" + str(date.today()) + "_" + str(NUM_STEPS) + "_" + str(NUM_EPOCH) + ".pt")
    else: 
        if not os.path.exists("./model/model_weights"):
            os.makedirs("./model/model_weights")
        torch.save(model.state_dict(), "./model/model_weights/model_" + str(date.today()) + "_" + str(NUM_STEPS) + "_" + str(NUM_EPOCH) + ".pt")

