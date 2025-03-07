import torch.nn as nn
class ConnectionLoss(nn.Module):
    def __init__(self):
        super(ConnectionLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, labels):
        """
        计算二分类交叉熵损失
        :param predictions: 模型预测的边标签
        :param labels: GT 的边标签 (0 或 1)
        """
        return self.bce_loss(predictions, labels)