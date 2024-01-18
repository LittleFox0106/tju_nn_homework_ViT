import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange

# 设置随机种子以保证实验的可重复性
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=2)

# 滤波器数量
num_filters_conv1 = 256
num_filters_conv2 = 512
#num_filters_conv3 = 1024  # 新添加的卷积层的滤波器数量

# 定义两层卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters_conv1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters_conv1, num_filters_conv2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(num_filters_conv2 * 8 * 8, 512)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, num_filters_conv2 * 8 * 8)
        x = self.relu(self.fc1(x))
        return x

# 定义三层卷积神经网络模型
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, num_filters_conv1, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(num_filters_conv1, num_filters_conv2, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(num_filters_conv2, num_filters_conv3, kernel_size=3, padding=1)  # 新添加的卷积层
#         self.fc1 = nn.Linear(num_filters_conv3 * 8 * 8, 512)

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = self.pool(self.relu(self.conv3(x)))  # 使用新添加的卷积层
#         x = x.view(-1, num_filters_conv3 * 8 * 8)
#         x = self.relu(self.fc1(x))
#         return x

# 定义视觉Transformer模块
class VisionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes):
        super(VisionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim),
            num_layers
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # 添加这行代码，解决2层神经网络里的维度问题，但是解决不了三层的
        #print("Before rearrange:", x.shape)
        x = rearrange(x, 'b n d -> n b d')  # 调整输入形状以适应Transformer
        #print("After rearrange:", x.shape)
        x = self.transformer(x)
        x = rearrange(x, 'n b d -> b n d')
        x = x.mean(dim=1)  # 取平均以获得整个序列的表示
        x = self.fc(x)
        return x

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNN().to(device)
transformer_model = VisionTransformer(input_dim=512, hidden_dim=256, num_heads=8, num_layers=4, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(cnn_model.parameters()) + list(transformer_model.parameters()), lr=0.001)

# 训练网络
num_epochs = 10
for epoch in range(num_epochs):
    cnn_model.train()
    transformer_model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 使用卷积神经网络提取特征
        cnn_outputs = cnn_model(inputs)

        # 使用视觉Transformer处理提取的特征
        transformer_outputs = transformer_model(cnn_outputs)

        # 计算损失
        loss = criterion(transformer_outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:  # 每200个batch输出一次
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

# 测试网络
cnn_model.eval()
transformer_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # 使用卷积神经网络提取特征
        cnn_outputs = cnn_model(images)

        # 使用视觉Transformer处理提取的特征
        transformer_outputs = transformer_model(cnn_outputs)

        _, predicted = torch.max(transformer_outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on the test images: %d %%' % (100 * correct / total))