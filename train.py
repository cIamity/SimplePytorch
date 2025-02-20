from torch.utils.tensorboard import SummaryWriter
import torch, torchvision
from torch.nn import CrossEntropyLoss
from core import tool

# 初始化TensorBoard
writer = SummaryWriter("log")

#region 准备数据
transformer = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.ToTensor(),
    ])

## 加载数据  CIFAR10
data_root = 'F:\\TORCH_HOME\\dataset\\CIFAR10'          # x: (batch_size, 3, 32, 32) -> y: (batch_size)
train_data = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transformer)
test_data = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transformer)
train_dataloader = torch.utils.data.DataLoader(train_data, 128, True, drop_last = True)
test_dataloader = torch.utils.data.DataLoader(test_data, 128, True)
#endregion

#region 准备训练
# 加载模型
model = tool.load_model("best_model.pt")
print(model)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=20)
criterion = CrossEntropyLoss()

# 早停参数
patience = 5  
best_loss = float('inf')
wait = 0  # 记录未提升的轮数
#endregion

best_acc = 0.0
# 训练
for epoch in range(100):
    train_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  
        outputs = model(images)[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_loss = train_loss / len(train_dataloader)

    writer.add_scalar('accuaracy/train', train_acc, epoch)
    writer.add_scalar('Loss/train', avg_loss, epoch)

    # 在测试集上评估
    test_loss = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)[0]
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    avg_test_loss = test_loss / len(test_dataloader)

    writer.add_scalar('accuaracy/test', test_acc, epoch)
    writer.add_scalar('Loss/test', avg_test_loss, epoch)

    # 学习率调度器
    scheduler.step()

    # 记录最佳模准确率
    if test_acc > best_acc:
        best_acc = test_acc

    print(f"Epoch [{epoch+1}] - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # 早停逻辑
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        wait = 0  # 重置等待计数
    else:
        wait += 1
        if wait >= patience:
            print("Early Stopping!")
            break  # 提前终止训练

print("训练完成！最佳测试集准确率:", best_acc)


writer.close()

# 是否保存模型
save_option = input("是否保存模型？(y/n): ").strip().lower()

if save_option == "y":
    torch.save(model, "best_model.pt") 
    print("模型已成功保存为 best_model.pt")
