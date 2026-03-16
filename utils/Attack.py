import torch
import torch.nn as nn
import torch.nn.functional as F


class AWP:
    def __init__(self, model, optimizer, *, adv_param='weight',
                 adv_lr=0.001, adv_eps=0.001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def perturb(self):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()  # save model parameters
        self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])


# -----------------------------------AWP训练-----------------------------------
""" 
# 初始化AWP
awp = AWP(model, optimizer)

for step, batch in enumerate(train_loader):
    inputs, labels = batch
    optimizer.zero_grad()
    # forward + backward + optimize
    predicts = model(inputs)          # 前向传播计算预测值
    if epoch >= awp_start:
        awp.perturb()
    loss = loss_fn(predicts, labels)  # 计算当前损失
    loss.backward()       # 反向传播计算梯度
    awp.restore()
    optimizer.step()

"""

class FGSM:
    def __init__(self, model, epsilon=0.001):
        self.model = model
        self.epsilon = epsilon

    def attack(self, inputs, labels):
        inputs.requires_grad = True
        _, outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        data_grad = inputs.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_data = inputs + self.epsilon * sign_data_grad
        perturbed_data = torch.clamp(perturbed_data, 0, 1) # 保持输入数据在有效范围内
        return perturbed_data
    
# -----------------------------------FGSM训练-----------------------------------
""" 
# 初始化FGSM
fgsm = FGSM(model,epsilon=0.001)

for step, batch in enumerate(train_loader):
    inputs, labels = batch
    optimizer.zero_grad()

    # forward + backward + optimize
    predicts = model(inputs)          # 前向传播计算预测值
    loss = loss_fn(predicts, labels)  # 计算当前损失
    loss.backward()       # 反向传播计算梯度
    optimizer.step()

    # FGSM attack
    if epoch >= 50:
        optimizer.zero_grad()
        perturbed_data = fgsm.attack(inputs.float(), labels.long())   
        fea, out = model(perturbed_data)
        attack_loss = clf_loss_func(out, labels.long())
        loss = attack_loss
        loss.backward()
        optimizer.step()

"""