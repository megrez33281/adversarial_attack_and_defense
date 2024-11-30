import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,datasets
import Original_model
import Defense_model
from Get_test import get_device, get_test_loader


def mifgsm_attack(input,epsilon,data_grad):
  iter=10   # 跌代次數
  decay_factor=1.0 # 先前gradient佔的比重
  pert_out = input  # X*
  alpha = epsilon/iter # stride(步幅)
  gradient = 0  # 梯度向量
  for i in range(iter-1):
    gradient = decay_factor*gradient + data_grad/torch.norm(data_grad,p=1)
    pert_out = pert_out + alpha*torch.sign(gradient) # alpha：步幅，torch.sign(gradient)：方向
    pert_out = torch.clamp(pert_out, 0, 1) # 限制pert_out中小於0的元素=0，大於1的元素=1
    if torch.norm((pert_out-input),p=float('inf')) > epsilon:
      # 若原先的input與加入擾動的pert_out的L∞ norm > epsilon 便停止
      break
  return pert_out


def attack_original_model(model, device, test_loader, epsilon):
    correct = 0
    adversarial_examples = []
    for data, target in test_loader:

        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        # put data into model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] 
        if init_pred.item() != target.item():
            # original sample predict error
            # the original sample which predict error won't count
            continue
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()   
        data_grad = data.grad.data
        # compute perturbed sample X*
        perturbed_data = mifgsm_attack(data, epsilon, data_grad)

        # predict X*
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            # X* predict correctly
            correct += 1
            if (epsilon == 0) and (len(adversarial_examples) < 5):
                # 為epsilon = 0 蒐集adversial_example（epsilon = 0等於不加入擾動，其結果等同於原始輸入）
                # 確保epsilon = 0 時adversarial_examples不會是空的
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adversarial_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # 蒐集加擾動後使model預測失敗的adversial example
            if len(adversarial_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adversarial_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    return final_acc, adversarial_examples

def attack_distillation_model(model, device, test_loader, epsilon):
    correct = 0
    adversarial_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        # 將output輸入激活函數
        output = F.log_softmax(output,dim=1)
        init_pred = output.max(1, keepdim=True)[1] 
        if init_pred.item() != target.item():
            # 原始樣本預測錯誤
            # 不計入原先就預測錯誤的樣本
            continue
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()   
        data_grad = data.grad.data
        # 計算加入擾動的X*
        perturbed_data = mifgsm_attack(data, epsilon, data_grad)

        # 測試加入擾動的X*
        output = model(perturbed_data)
        output = F.log_softmax(output,dim=1)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            # 加入擾動後仍預測正確
            correct += 1
            if (epsilon == 0) and (len(adversarial_examples) < 5):
                # 為epsilon = 0 蒐集adversial_example（epsilon = 0等於不加入擾動，其結果等同於原始輸入）
                # 確保epsilon = 0 時adversarial_examples不會是空的
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adversarial_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # 蒐集加擾動後使model預測失敗的adversial example
            if len(adversarial_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adversarial_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    return final_acc, adversarial_examples


def test_attack_original_model(model, device):
    print("attack original model......")
    epsilons = [0,0.007,0.01,0.02,0.03,0.05,0.1,0.2,0.3]
    accuracies = []
    adversarial_examples = []
    
    test_loader, label_names = get_test_loader()
    
    for eps in epsilons:
        acc, ex = attack_original_model(model, device, test_loader, eps)
        accuracies.append(acc)
        adversarial_examples.append(ex)
    print()
    # draw accuracy under different epsilons
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.title("MI-FGSM")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    # draw adversarial example
    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(adversarial_examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(adversarial_examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = adversarial_examples[i][j]
            plt.title("{} -> {}".format(label_names[orig], label_names[adv]))
            if len(np.shape(ex)) == 3:
                plt.imshow(np.moveaxis(ex, 0, -1))
            else:
                plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()

def test_attack_distillation_model(model, device):
    print("attack distillation model......")
    epsilons = [0,0.007,0.01,0.02,0.03,0.05,0.1,0.2,0.3]
    accuracies = []
    adversarial_examples = []
    
    test_loader, label_names = get_test_loader()
    for eps in epsilons:
        acc, ex = attack_distillation_model(model, device, test_loader, eps)
        accuracies.append(acc)
        adversarial_examples.append(ex)
    print()
    # 畫出不同epsilons下model的accuracy
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.title("MI-FGSM")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    # 畫出各adversarial example
    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(adversarial_examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(adversarial_examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = adversarial_examples[i][j]
            plt.title("{} -> {}".format(label_names[orig], label_names[adv]))
            if len(np.shape(ex)) == 3:
                plt.imshow(np.moveaxis(ex, 0, -1))
            else:
                plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
    
    

if __name__ == '__main__':
    device = get_device()
    model = Original_model.read_model(device)
    test_attack_original_model(model, device)
    modelF1 = Defense_model.read_model(device)
    test_attack_distillation_model(modelF1, device)

   