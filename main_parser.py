import argparse
from torchsummary import summary

#from layers_basic import *
from model_basic import *
from train import *
from data.data import load_datasets

parser = argparse.ArgumentParser(description='Create sample')

parser.add_argument("--img_size", type=int, default=64)

parser.add_argument("--state", type=str, default="idle") #train, test

parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--n_epochs", type=int, default=0)

parser.add_argument("--num_image_channels", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--feature_map_sizes", type=list, default=[64, 64, 64])
parser.add_argument("--filter_sizes", type=list, default=[5, 5, 5])

parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--save_path", type=str, default="")

args = parser.parse_args()

print("State :", args.state)

train_dataset, test_dataset, train_dataloader, test_dataloader = load_datasets()

print(len(test_dataset))

#Size to be changed
img_size = args.img_size
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
# Images are gray-scale, so we only have one image channel
num_image_channels = args.num_image_channels
# Number of classes
num_classes = args.num_classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

batch_size = args.batch_size

# create instance of our model
feature_map_sizes = args.feature_map_sizes
filter_sizes = args.filter_sizes


model = ConvNet(feature_map_sizes, filter_sizes, activation=torch.nn.ReLU()).double()
if args.model_path != "": 
    model.load_state_dict(torch.load("./model_weights/" + args.model_path, map_location=device))
    NUM_EPOCH = int(args.model_path.split("_")[-1].split(".")[0])
else: 
    model.apply(weights_init)
    NUM_EPOCH = 0
# put the model in the device memory
model = model.to(device) #else?



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

# Optimization operation: Adam 
learning_rate = args.learning_rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if args.state == "train":
    train(device, model, cross_entropy_loss, learning_rate, optimizer, args.n_epochs, train_dataloader, test_dataloader, args.save_path, NUM_EPOCH)

