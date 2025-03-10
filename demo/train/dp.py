import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset

# 1. 定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 2. 生成一些模拟数据
def generate_dummy_data(num_samples, input_size):
    X = torch.randn(num_samples, input_size)  # 输入数据
    y = torch.randint(0, 2, (num_samples,))  # 随机标签（二分类）
    return X, y

# 3. 主函数
def train_data_parallel():
    # 参数设置
    input_size = 10
    hidden_size = 20
    output_size = 2
    batch_size = 32
    num_samples = 1000
    num_epochs = 5

    # 检查可用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, GPU count: {torch.cuda.device_count()}")

    # 生成数据
    X, y = generate_dummy_data(num_samples, input_size)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = SimpleNN(input_size, hidden_size, output_size)

    # 如果有多个 GPU，使用 DataParallel 包装模型 ， 单节点多卡
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for Data Parallelism!")
        model = DataParallel(model)
    
    # 将模型移动到 GPU
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            # 将数据移动到 GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training finished!")

# 4. 运行训练
# python deploy/modal/src/train/demo/dp.py
if __name__ == "__main__":
    train_data_parallel()