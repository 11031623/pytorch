import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ========== 1. 生成数据 ==========
X, y = make_friedman1(
    n_samples=1500,
    n_features=10,
    noise=1.0,
    random_state=42
)

# ========== 2. 数据处理 ==========
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 标准化输入特征（非常重要！）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ========== 3. 定义线性模型 ==========
class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # 输入 10 维 → 输出 1 维

    def forward(self, x):
        return self.linear(x)

model = LinearModel(input_dim=10)

# ========== 4. 定义损失函数和优化器 ==========
criterion = nn.MSELoss()  # 回归任务用 MSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ========== 5. 训练 ==========
num_epochs = 500
train_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    y_pred = model(X_train)

    # 计算损失
    loss = criterion(y_pred, y_train)

    # 反向传播
    loss.backward()
    #更新参数
    optimizer.step()

    train_losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ========== 6. 测试 ==========
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    test_loss = criterion(y_pred_test, y_test)
    print(f"\n最终测试 MSE: {test_loss.item():.4f}")

# 计算 R² 分数
from sklearn.metrics import r2_score
r2 = r2_score(y_test.numpy(), y_pred_test.numpy())
print(f"测试 R² 分数: {r2:.4f}")

# ========== 7. 可视化 ==========
# 图 1：训练损失曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid(True)

# 图 2：预测值 vs 真实值散点图
plt.subplot(1, 2, 2)
plt.scatter(y_test.numpy(), y_pred_test.numpy(), alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title(f"Prediction vs True (R² = {r2:.4f})")
plt.grid(True)

plt.tight_layout()
plt.savefig("linear_model_result.png", dpi=150)
plt.show()

# ========== 8. 打印模型结构 ==========
print("\n模型结构：")
print(model)