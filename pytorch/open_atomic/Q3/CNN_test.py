import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ============================================================
# 模型定义（训练和测试都用这一个）
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 43)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================
# 基础变换（训练 + 原始测试）
# ============================================================
base_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

MODEL_PATH = "cnn_model1.pth"
BATCH_SIZE = 64


# ============================================================
# 训练函数（只跑一次）
# ============================================================
def train_model():
    trainset = torchvision.datasets.GTSRB(
        root="./data", split="train", download=True, transform=base_transform
    )
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存到 {MODEL_PATH}")


# ============================================================
# 定义多种测试变换
# ============================================================
test_transforms = {
    "原始": base_transform,

    "变亮": transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),

    "模糊": transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.GaussianBlur(kernel_size=5, sigma=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),

    "旋转": transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),

    "遮挡": transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, scale=(0.02, 0.1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
}


# ============================================================
# 测试函数
# ============================================================
def test_with_transform(model, transform, name):
    testset = torchvision.datasets.GTSRB(
        root="./data", split="test", download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    print(f"{name}: {acc:.2f}%")
    return acc


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # 第一步：如果模型不存在，先训练
    if not os.path.exists(MODEL_PATH):
        print("未找到模型，开始训练...")
        train_model()
    else:
        print(f"已找到模型 {MODEL_PATH}，跳过训练。")

    # 第二步：加载模型
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 第三步：测试所有变换
    results = {}
    for name, transform in test_transforms.items():
        acc = test_with_transform(model, transform, name)
        results[name] = acc

    # 第四步：打印汇总结果
    print("\n" + "=" * 40)
    print("鲁棒性测试结果汇总")
    print("=" * 40)
    for name, acc in results.items():
        print(f"{name}: {acc:.2f}%")
