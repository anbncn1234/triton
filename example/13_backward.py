

import torch

# 1. 生成数据
x = torch.linspace(0, 100, steps=100, dtype=torch.float32)
y = x + torch.rand(100) * 10  # y = x + 噪声

# 2. 初始化可训练参数
a = torch.randn(1, requires_grad=True)  # 权重
b = torch.randn(1, requires_grad=True)  # 偏置

# 3. 定义学习率
learning_rate = 0.0000001

# 4. 训练 1000 次
for epoch in range(1000):
    # 前向传播
    f = a * x + b
    loss = torch.sum((y - f) ** 2)  # MSE Loss

    # 反向传播（自动计算梯度）
    #loss.backward()
    #manual_grad_a = -2 * torch.sum(x * (y - (a.detach() * x + b.detach())))
    #manual_grad_b = -2 * torch.sum(y - (a.detach() * x + b.detach()))
    # 手动更新参数（梯度下降）
    grad_a = -2 * torch.sum(x * (y - f))  # dL/da
    grad_b = -2 * torch.sum(y - f)         # dL/db
    
    # 手动更新参数
    a =a - learning_rate * grad_a
    b =a -  learning_rate * grad_b
        
    #with torch.no_grad():  # 禁用梯度跟踪
        #a -= learning_rate * a.grad
        #b -= learning_rate * b.grad
        
        # 清零梯度（重要！）
        #a.grad.zero_()
        #b.grad.zero_()

    # 每 100 次打印损失
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 5. 打印最终参数
print("\nFinal Parameters:")
print("a:", a.item())
print("b:", b.item())