import argparse
from torchsummary import summary

#from layers_basic import *
from model.model import *
from model.train import *
from data.data import load_datasets

parser = argparse.ArgumentParser(description='Create sample')

parser.add_argument("--img_size", type=int, default=64)

parser.add_argument("--state", type=str, default="idle") #train, test

parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--decay", type=float, default=0.95)
parser.add_argument("--n_epochs", type=int, default=0)
parser.add_argument("--use_sampler", type=bool, default=True)

parser.add_argument("--num_image_channels", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--feature_map_sizes", type=list, default=[64, 64, 64, 64])
parser.add_argument("--filter_sizes", type=list, default=[5, 5, 5, 5])
parser.add_argument("--hidden", type=list, default=[128, 10, 4]) #Always keep 4

parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--save_path", type=str, default="")
parser.add_argument("--keep", type=bool, default=False)

args = parser.parse_args()

print("State :", args.state)

train_dataset, test_dataset, train_dataloader, test_dataloader = load_datasets(batch_size = args.batch_size, img_size = args.img_size, use_sampler=args.use_sampler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

#TODO we are using padding = same for now so the img size remains the same after the conv layer, only the maxpool /2 the size

model = ConvNet(args.feature_map_sizes, args.filter_sizes, args.hidden, args.num_classes, args.img_size, activation=torch.nn.LeakyReLU()).double()
model = model.to(device)

# Optimization operation: Adam 
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay, verbose=True)

if args.model_path != "": 
    if os.path.isfile("./model/model_weights/" + args.model_path):
        checkpoint = torch.load("./model/model_weights/" + args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        scheduler = checkpoint['scheduler']
        optimizer.load_state_dict(checkpoint['optimizer'])
        NUM_EPOCH = checkpoint['epoch']
        NUM_STEPS = checkpoint['step']
        print(NUM_EPOCH)
else: 
    model.apply(weights_init)
    NUM_EPOCH = 0
    NUM_STEPS = 0
# put the model in the device memory

#TODO: weights initialisation

#TODO: make this work, pb between float and double
#summary(model.float(), input_size=(num_image_channels, img_size, img_size))

# count total number of parameters including non trainable
total_params_count = sum(p.numel() for p in model.parameters())
# count total trainable parameters
trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total number of total parameters: {total_params_count}")
print(f"Number of trainable parameters: {trainable_params_count}")

# Loss: Cross-Entropy
cross_entropy_loss = torch.nn.CrossEntropyLoss()
#TODO use scheduler

if args.state == "train":
    train(device, model, cross_entropy_loss, args.learning_rate, optimizer, scheduler,  args.n_epochs, train_dataloader, test_dataloader, args.save_path, NUM_EPOCH, NUM_STEPS, args.batch_size, args.keep)

#Not a real test just to check some results
if args.state == "test":
    test_loss, test_accuracy = evaluate(device, model, cross_entropy_loss, args.learning_rate, optimizer, test_dataloader)
    print(f"Validation : accuracy = {test_accuracy}, loss = {test_loss}")