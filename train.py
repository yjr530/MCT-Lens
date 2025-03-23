import torch.nn as nn
from torchvision import transforms, datasets, models
import json
import os
import torch.optim as optim
from resnet_vit import vit_base
import torch
from torch.autograd import Variable
from ConfusionMatrix import ConfusionMatrix
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def string_rename(old_string, new_string, start, end):
    new_string = old_string[:start] + new_string + old_string[end:]
    return new_string

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# get data root path
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  

image_path = "./image_experiment/"  # lens data set path

pretrained_path='checkpoint/moco.ckpt'

train_dataset = datasets.ImageFolder(root=image_path + "train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# {'non-lens':0, 'lens':1}
print(train_dataset.classes)

test_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in test_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)


validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)

validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

print("\n==========  CNN-Transformer  ==========")
model_name = "CNN_Transformer"

net=vit_base(2,2)
print(net)
checkpoint = torch.load(pretrained_path)

net_dict = net.state_dict()

for k in net_dict.keys():
    print(k + '   ' + str(list(net_dict[k].size())))

new_moco_model = {}
for k, v in checkpoint['state_dict'].items():
    new_k = '.'.join(k.split('.')[1:])  
    new_moco_model[new_k] = v
    
pretrained_dict = {k: v for k, v in new_moco_model.items() if (k in net_dict.keys())}
      

for k, v in checkpoint.items():
     if isinstance(v, torch.Tensor):
         print(k + '   ' + str(list(v.size())))
     else:
         print(k + '   ' + str(v))
    

net_dict.update(pretrained_dict)

net.load_state_dict(net_dict)

net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)

best_auc = 0.0
save_path = './{}_64_lr4_10.pth'.format(model_name)


train_loss_list = []
val_loss_list = []

for epoch in range(200):
    # train
    net.train()
    running_loss = 0.0

    print("\n========== Epoch {} ==========".format(epoch + 1))
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        
        scheduler.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
 
    train_loss = running_loss / len(train_loader)
    train_loss_list.append(train_loss)
    print()

    # validate
    net.eval()
    val_running_loss = 0.0
    acc = 0.0  # accumulate accurate number / epoch
    predict_y_list = []  
    true_y_list = []  
    # p_a=p_b=n_a=n_b=0.0
    with torch.no_grad():
        label = ['non-lens', 'lens']
        confusion = ConfusionMatrix(num_classes=2, labels=label)

        #        print(p_a, p_b, n_a, n_b)
        for val_data in validate_loader:
            val_images, val_labels = val_data
            optimizer.zero_grad()
            outputs = net(val_images.to(device))
            
            loss = loss_function(outputs, val_labels.to(device))
            val_running_loss += loss.item()
            
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
            confusion.update(predict_y.cpu().numpy(), val_labels.cpu().numpy())
            
            predict_prob = torch.softmax(outputs, dim=1)[:, 1]
            predict_y_list.append(predict_prob.cpu().numpy())
            true_y_list.append(val_labels.cpu().numpy())

        val_accurate = acc / val_num
        predict_y_all = np.concatenate(predict_y_list)  
        true_y_all = np.concatenate(true_y_list)  
        auc = roc_auc_score(true_y_all, predict_y_all)  
        
        
        val_loss = val_running_loss / len(validate_loader)
        val_loss_list.append(val_loss)

        print("val_accurate:", val_accurate)
        print("AUC:", auc)
        if auc > best_auc:
            best_auc = auc
            torch.save(net.state_dict(), save_path)

            confusion.summary()
        
        print(f'Current best AUC: {best_auc}')
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f  P: %.3f  R: %.3f  F1: %.3f' %
               (epoch + 1, running_loss / step, val_accurate,p,r,f))

plt.plot(range(1, len(train_loss_list)+1), train_loss_list, label='Training')
plt.plot(range(1, len(val_loss_list)+1), val_loss_list, label='Validation')
plt.xlabel('Training Epoch')
plt.ylabel('Cross Entropy')
plt.title('Loss Curve')
plt.legend()
plt.show()

print('Finished Training')