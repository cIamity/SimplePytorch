from core import tool
import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



# 打开主窗口
tool.start()



# 模型准确性验证
data_root = 'F:\\TORCH_HOME\\dataset\\CIFAR10'

transformer = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    ])

test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transformer)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)  # 不打乱数据顺序

model = tool.load_model("best_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU（如果可用）
model.to(device)

correct = 0
total = 0

model.eval()
with torch.no_grad():  
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)[0]
        _, predicted = torch.max(outputs, 1)  
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算准确率
accuracy = correct / total
print(f"模型在测试集上的准确率: {accuracy:.2f}")