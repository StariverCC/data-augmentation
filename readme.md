# 说明文档

### 运行说明

#### 1、下载依赖

##### pip 安装`pip install -r requirements.txt`

##### conda安装 `conda install --file requirements.txt`

#### 2、运行程序

`python ./compare.py`

### 程序说明

#### 1、数据集

##### 本实验使用cifar10数据集作为训练数据

CIFAR-10 是计算机视觉领域广泛使用的一个基准数据集。它包含了 60,000 张 32x32 像素的彩色图像，这些图像共分为 10 个类别，每个类别有 6,000 张图像。这些类别分别是：飞机、汽车、鸟、猫、鹿、狗、蛙、马、船和卡车。数据集被进一步分为 50,000 张训练图像和 10,000 张测试图像，训练集和测试集都是均匀分布的，即每个类别在训练集中有 5,000 张图像，在测试集中有 1,000 张图像。

数据集导入代码如下

##### 数据集划分

CIFAR-10 数据集的划分方式是预先定义好的，它包括 50,000 张训练图像和 10,000 张测试图像。

- **训练集**：包含 50,000 张图像，用于训练模型。这些图像均匀分布在 10 个类别中，每个类别有 5,000 张图像。训练集用于训练模型，使模型能够学习到从输入图像到输出标签（类别）的映射。
- **测试集**：包含 10,000 张图像，用于评估模型的性能。测试集的图像也均匀分布在 10 个类别中，每个类别有 1,000 张图像。

##### 模型导入

导入模型的过程中会自动下载模型

```python
train_dataset_basic = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_basic)
train_dataset_augmented = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_augmented)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_basic)

train_loader_basic = DataLoader(train_dataset_basic, batch_size=64, shuffle=True)
train_loader_augmented = DataLoader(train_dataset_augmented, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

#### 2、cnn网络定义

本实验参考了github上一个项目的网络定义

> https://github.com/MaxChanger/pytorch-cifar/blob/master/main_DaiNet9.py

网络是为了处理 CIFAR-10 数据集而设计的，CIFAR-10 数据集包含了 32x32 像素的彩色图像。网络通过一系列的卷积层、激活层、批量归一化层、池化层和全连接层对这些图像进行分类。

##### Layer 1

- **卷积层** (`nn.Conv2d(3, 64, kernel_size=3, padding=1)`)：输入通道为 3（因为 CIFAR-10 图像是彩色的，包含 RGB 三个通道），输出通道为 64，卷积核大小为 3x3，padding 设为 1 以保持图像尺寸不变。这一层的目的是提取图像的初级特征。
- **激活层** (`nn.ReLU()`)：ReLU 激活函数用于增加网络的非线性，没有它网络很容易退化成一个线性模型。
- **批量归一化层** (`nn.BatchNorm2d(64)`)：加速收敛，减少模型训练过程中的内部协变量偏移。

##### Layer 2

与 Layer 1 类似，但这一层使用相同数量的输出通道。它包含一个额外的最大池化层 (`nn.MaxPool2d(2, 2)`) 来降低特征图的维度，减少参数数量，提高计算效率。

##### Layer 3

- **卷积层** (`nn.Conv2d(64, 128, kernel_size=3, padding=1)`)：提高输出通道至 128，以捕获更复杂的特征。

##### Layer 4

此层再次应用最大池化来降低特征图的维度，并使用 Dropout（0.5）进行更强的正则化以避免过拟合。

##### Layer 5

卷积层的输出通道增至 256，通过卷积、激活和批量归一化层进一步提取特征。

##### Layer 6

最后一个卷积层继续保持 256 输出通道，并以平均池化层结束（`nn.AvgPool2d(8, 8)`），这有助于减少每个特征图的空间维度至 1x1，为全连接层准备。

##### 全连接层 (Fully Connected Layers)

**`self.fc1 = nn.Linear(1 \* 1 \* 256, 10)`**：因为 Layer 6 的平均池化输出是 1x1x256 的特征图，所以全连接层的输入特征数是 256。输出特征数为 10，对应 CIFAR-10 的十个类别。

##### 前向传播 (`forward` 方法)

`forward` 方法定义了数据通过网络的方式。它首先通过定义好的层序列处理输入 `x`，

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()     
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(8, 8)
        )
        self.fc1 = nn.Linear(1 * 1 * 256, 10)   # 接着三个全连接层 Linear(in_features, out_features, bias=True)
        self.fc2 = nn.Linear(128, 100) #                      输入样本的大小 输出样本的大小 若设置为False这层不学习偏置 默认True
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(-1, 1 * 1 * 256)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
                                    #  第一个参数-1是说这个参数由另一个参数确定， 比如矩阵在元素总数一定的情况下，确定列数就能确定行数。
                                    #  那么为什么这里只关心列数不关心行数呢，因为马上就要进入全连接层了，而全连接层说白了就是矩阵乘法，
                                    #  你会发现第一个全连接层的首参数是16*5*5，所以要保证能够相乘，在矩阵乘法之前就要把x调到正确的size
        x = self.fc1(x)

        return x
```

#### 3、数据增强方法

本实验使用了传统的针对图像的增强方法包括随机裁剪、随机水平翻转、随机旋转、色彩抖动、以及Mixup 数据增强技术

- 传统方法

​	主要使用一下方法对数据集进行处理

```
transform_augmented = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
    # transforms.RandomRotation(15),  # 随机旋转±15度
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 色彩抖动
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])
```

- mixup方法

  Mixup 通过在像素级别上混合两个图像及其对应标签来生成新的训练样本。
  
  
  $$
  给定两个随机的训练样本 (x 
  i
  ​
   ,y 
  i
  ​
   )和 
  
  (x 
  j
  ​
   ,y 
  j
  ​
   )，其中 
  
  x 是图像数据，
  y 是相应的标签。\\Mixup 通过以下方式生成新的训练样本 
  
  (x 
  ′
   ,y 
  ′
   )：x 
  ′
   =λx 
  i
  ​
   +(1−λ)x 
  j
  ​
   
  
  ′
  =
  
  +
  (
  1
  −
  
  )
  
  y 
  ′
   =λy 
  i
  ​
   +(1−λ)y 
  j
  ​
   \\
  这里，
  λ 是从某个分布（通常是 Beta 分布）中抽取的一个值，范围在 0 到 1 之间。\\通过这种方式，Mixup 创建的新图像是两个原始图像的加权平均，其标签也是相应地进行加权平均得到的。
  $$
  
  
  ```python
  def mixup_data(x, y, alpha=0.5, device='gpu'):
      """
      Applies MixUp augmentation to a batch of data.
      
      Parameters:
      - x: 输入数据
      - y: 对应的标签
      - alpha: 控制Beta分布的参数，影响混合比例
      - device: 指定数据处理的设备（'gpu' 或 'cpu'）
      
      Returns:
      - mixed_x: 混合后的输入数据
      - y_a: 原始标签
      - y_b: 与原始数据混合的数据的标签
      - lam: 用于混合的比例
      """
      # 如果alpha大于0，则从Beta分布中抽取lam，否则设置lam为1
      if alpha > 0:
          lam = np.random.beta(alpha, alpha)
      else:
          lam = 1
      
      # 获取批次大小
      batch_size = x.size()[0]
      # 生成一个随机排列，用于选择混合的样本
      index = torch.randperm(batch_size).to(device)
  
      # 根据lam混合输入数据
      mixed_x = lam * x + (1 - lam) * x[index, :]
      # 选择对应的标签
      y_a, y_b = y, y[index]
      
      return mixed_x, y_a, y_b, lam
  
  
  def mixup_criterion(criterion, pred, y_a, y_b, lam):
      """
      Computes the MixUp criterion.
      
      Parameters:
      - criterion: 损失函数
      - pred: 模型预测结果
      - y_a: 原始标签
      - y_b: 混合标签
      - lam: 混合比例
      
      Returns:
      - 混合损失
      """
      # 计算并返回基于混合比例lam加权的损失
      return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
  
  ```

#### 4、实验评估

​	本实验通过设置三个方面的评估标准来，评估数据增强前后模型的表现，分别是acc准确率评估、整体precision评估、每个分类的precision评估。

- baseline设置

  通过在模型导入过程中不进行任何数据增强操作

- acc准确率评估

  针对模型的train_acc和test_acc在每个epoch中的表现进行评估

- 整体precision评估

  针对模型在每一个epoch中测试集上的precision表现进行评估

- 每个分类的precision评估

  针对模型最终在测试集上的每个分类的precision进行评估

- precison部分代码

  ```python
  def evaluate_precision(model, test_loader, device):
      model.eval()  # Set model to evaluation mode
      all_predictions = []
      all_targets = []
  
      with torch.no_grad():
          for inputs, labels in test_loader:
              inputs = inputs.to(device)
              labels = labels.to(device)
              outputs = model(inputs)
              _, predicted = torch.max(outputs, 1)
              all_predictions.extend(predicted.cpu().numpy())
              all_targets.extend(labels.cpu().numpy())
  
      # Calculate precision for each class, with zero_division parameter set to 0
      precision_per_class = precision_score(all_targets, all_predictions, labels=range(10), average=None, zero_division=0)
      overall_precision = precision_score(all_targets, all_predictions, labels=range(10), average='macro', zero_division=0)
      return precision_per_class, overall_precision
  ```

- 训练代码

  在训练中加入参数c，来表示是否使用了mixup数据增强方法，其余传统方法通过注释的方法进行实验

  ```python
  def train_and_evaluate(model, train_loader, test_loader, epochs=5,c=0):
      overall_precision_list=[]
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model.to(device)
      optimizer = optim.Adam(model.parameters(), lr=0.001)
      criterion = nn.CrossEntropyLoss()
  
      train_acc_history, test_acc_history = [], []
  
      for epoch in range(epochs):
          correct_train, total_train = 0, 0
  
          for images, labels in train_loader:
              images, labels = images.to(device), labels.to(device)
              if c==1:
                  # Apply MixUp
                  mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0, device=device)
                  optimizer.zero_grad()
                  outputs = model(mixed_images)
                  # Adjust loss calculation for MixUp
                  loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                  loss.backward()
                  optimizer.step()
              else:
                  optimizer.zero_grad()  # 清空优化器的梯度缓存
                  outputs = model(images)  # 通过模型前向传播得到输出
                  loss = criterion(outputs, labels)  # 计算损失
                  loss.backward()  # 反向传播，计算梯度
                  optimizer.step()  # 使用计算得到的梯度更新模型参数
              # Calculate accuracy with original data, not mixed, for simplicity
              _, predicted = torch.max(outputs.data, 1)
              total_train += labels.size(0)
              correct_train += (predicted == labels).sum().item()
  
  
          train_acc = correct_train / total_train
          train_acc_history.append(train_acc)
  
          model.eval()
          correct_test, total_test = 0, 0
          with torch.no_grad():
              for images, labels in test_loader:
                  images, labels = images.to(device), labels.to(device)
                  outputs = model(images)
                  _, predicted = torch.max(outputs.data, 1)
                  total_test += labels.size(0)
                  correct_test += (predicted == labels).sum().item()
  
          test_acc = correct_test / total_test
          test_acc_history.append(test_acc)
  
          precision_per_class, overall_precision = evaluate_precision(model, test_loader, device)
          overall_precision_list.append(overall_precision)
          print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Overall Precision: {overall_precision}')
          print(f'Precision per class: {precision_per_class}')
          print(f'Epoch [{epoch + 1}/{epochs}], Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
      return train_acc_history, test_acc_history,precision_per_class,overall_precision_list
  ```

#### 5、实验结果

- 使用mixup数据增强的结果

  - acc准确率评估![image-20240412021859251](image-20240412021859251.png)

  - 整体precision评估

    ![image-20240412022029655](image-20240412022029655.png)

  - 每个分类的precision评估

    ![image-20240412022043912](image-20240412022043912.png)

- 使用随机裁剪和随机水平翻转的结果

  - acc准确率评估

    ![image-20240412022858247](image-20240412022858247.png)

  - 整体precision评估

    ![image-20240412023144356](image-20240412023144356.png)

  - 每个分类的precision评估

    ![image-20240412023156639](image-20240412023156639.png)

- 使用随机旋转和色彩抖动的结果

  - acc准确率评估

  ![image-20240412023236861](image-20240412023236861.png)

  - 整体precision评估

  ![image-20240412023248139](image-20240412023248139.png)

  - 每个分类的precision评估

    ![image-20240412023258186](image-20240412023258186.png)

- 使用四种传统增强手段的结果

  - acc准确率评估

    ![image-20240412023318915](image-20240412023318915.png)

  - 整体precision评估

    ![image-20240412023329758](image-20240412023329758.png)

  - 每个分类的precision评估

    ![image-20240412023359466](image-20240412023359466.png)

​		

