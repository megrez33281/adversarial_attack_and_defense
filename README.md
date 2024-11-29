# Reference
* code 
    https://github.com/as791/Adversarial-Example-Attack-and-Defense

* Momentum Iterative Fast Gradient Sign Method(MI-FGSM)
    https://arxiv.org/abs/1710.06081

* Defensive Distillation 
    https://arxiv.org/abs/1511.04508


# Set up virtual environment
```
python -m venv venv
venv\scripts\activate
```

# install package
```
pip install -r package.txt
```

# Illustrate

## Train Original Model
Run Original_model.py

## Train Defense Model
Run Defense_model.py

## MI-FGSM attack
Run mifgsm_attack.py
* Original_model.read_model(device)
    read Original Model

* test_attack_original_model(model, device)
    attack Original Model

* Defense_model.read_model(device)
    read Defense Model

* test_attack_distillation_model(model, device)
    attack Defense Model


# Result
## FashionMNIST
* original model architecture
    ```
    class Net(nn.Module):
    # neural network model
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    ```
    * original model loss  
        ![image](img/fashion_mnist/original_model_loss.png)


* defense (Distillation) model architecture
    ```
    class NetF(nn.Module):
        def __init__(self):
            super(NetF, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return x

    class NetF1(nn.Module):
        def __init__(self):
            super(NetF1, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, 1)
            self.conv2 = nn.Conv2d(16, 32, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(4608, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return x
    ```
    * defense modelF loss  
        ![image](img/fashion_mnist/defense_modelF_Loss.png)

    * defense modelF1 (train with soft label) loss  
        ![image](img/fashion_mnist/defense_modelF1_Loss.png)

* attack original model result
    * accuracy each epsilon  
        ![image](img/fashion_mnist/MI-FGSM_attack_original_model_accuracy.png)
    * result  
        ![image](img/fashion_mnist/MI-FGSM_attack_original_model.png)

* attack defense (Distillation) model result
    * accuracy each epsilon  
        ![image](img/fashion_mnist/MI-FGSM_attack_defense_model_accuracy.png)
    * result  
        ![image](img/fashion_mnist/MI-FGSM_attack_adversarial_model.png)


