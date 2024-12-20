import torch
from torch import nn
import torchvision
import tqdm
from torchinfo import summary
import preprocessing,engine
from pathlib import Path

torch.manual_seed(42)
torch.cuda.manual_seed(42)


device = "cuda" if torch.cuda.is_available() else "cpu"
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transforms = weights.transforms()
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

train_dir = './dataset/TRAIN/'
test_dir = './dataset/TEST/'

train_dataloader, test_dataloader, class_names = preprocessing.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=auto_transforms, # perform same data transforms on our own data as the pretrained model
                                                                               batch_size=32) # set mini-batch size to 32

output_shape = len(class_names)

for param in model.features.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Setup training and save the results
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=5,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Create target directory
target_dir_path = Path('models/')
target_dir_path.mkdir(parents=True,
                        exist_ok=True)

model_name = 'disease_classify_model.pth'

  # Create model save path
assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
model_save_path = target_dir_path / model_name

  # Save the model state_dict()
print(f"[INFO] Saving model to: {model_save_path}")
torch.save(obj=model.state_dict(),
             f=model_save_path)


