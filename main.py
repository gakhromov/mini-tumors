import argparse
from torchsummary import summary

# for the tensorboard summary writers
from torch.utils.tensorboard import SummaryWriter

#from layers_basic import *
from model.model import *
from model.train import *
from data.data import load_datasets

parser = argparse.ArgumentParser(description='Create sample')

parser.add_argument("--img_size", type=int, default=64)

parser.add_argument("--state", type=str, default="idle") #train, test

parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--n_epochs", type=int, default=0)

parser.add_argument("--num_image_channels", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--feature_map_sizes", type=list, default=[32, 64, 128, 256])
parser.add_argument("--filter_sizes", type=list, default=[3, 3, 3, 3])

parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--save_path", type=str, default="")

args = parser.parse_args()

print("State :", args.state)

train_dataset, test_dataset, train_dataloader, test_dataloader = load_datasets()

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

# Tensorboard
training_writer = SummaryWriter("./tensorboard/train")
validation_writer = SummaryWriter("./tensorboard/valid")

batch_size = args.batch_size

# create instance of our model
feature_map_sizes = args.feature_map_sizes
filter_sizes = args.filter_sizes

#TODO we are using padding = same for now so the img size remains the same after the conv layer, only the maxpool /2 the size

model = ConvNet(feature_map_sizes, filter_sizes, num_classes, img_size, activation=torch.nn.LeakyReLU()).double()
model = model.to(device)
if args.model_path != "": 
    model.load_state_dict(torch.load("./model_weights/" + args.model_path, map_location=device))
    temp = args.model_path.split("_")
    NUM_EPOCH = int(temp[-1].split(".")[0])
    NUM_STEPS = int(temp[-2].split(".")[0])
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
cross_entropy_loss = torch.nn.CrossEntropyLoss()
#TODO use scheduler

# Optimization operation: Adam 
learning_rate = args.learning_rate
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = ""#torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

if args.state == "train":
    train(device, model, cross_entropy_loss, learning_rate, optimizer, scheduler,  args.n_epochs, train_dataloader, test_dataloader, args.save_path, NUM_EPOCH, NUM_STEPS, training_writer, validation_writer)

#Not a real test just to check some results
if args.state == "test":
    test_loss, test_accuracy = evaluate(device, model, cross_entropy_loss, learning_rate, optimizer, test_dataloader, validation_writer)
    print(f"Validation : accuracy = {test_accuracy}, loss = {test_loss}")