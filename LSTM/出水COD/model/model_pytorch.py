import torch
import numpy as np
from torch.nn import Module, Linear, LSTM
from torch.utils.data import DataLoader, TensorDataset


class LSTMNet(Module):
    def __init__(self, config):
        super(LSTMNet, self).__init__()
        self.config = config

        self.lstm = LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_layers,
            batch_first=True,
            dropout=self.config.dropout_rate,
        )

        self.linear = Linear(
            in_features=self.config.lstm_hidden_size,
            out_features=self.config.ouput_size,
        )

    def forward(self, x, hidden=None):
        # LSTM前向计算
        lstm_out, hidden = self.lstm(x, hidden)
        # 获取LSTM输出的维度信息
        batch_size, time_step, hidden_size = lstm_out.shape

        if self.config.op == 0:
            # 将lstm_out变成(batch_size*time_step, hidden_size), 才能传入全连接层
            lstm_out = lstm_out.reshape(-1, hidden_size)
            # 全连接层
            linear_out = self.linear(lstm_out)
            # 转换维度, 用于输出
            output = linear_out.reshape(time_step, batch_size, -1)
            # 我们只需要返回最后一个时刻的数据即可
            return output[-1]

        elif self.config.op == 1:
            linear_out = self.linear(lstm_out)  # 全部时刻都输入到全连接层
            output = linear_out[:, -1, :]  # 取最后一个时刻的输出
            return output

        elif self.config.op == 2:
            linear_out = self.linear(lstm_out[:, -1, :])  # 取LSTM最后一个时刻的输出作为全连接层输入
            output = linear_out
            return output


def train(config, train_and_valid_data):
    model = LSTMNet(config).to(config.device)  # 模型实例化

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    # 先转为Tensor
    train_X, train_Y, valid_X, valid_Y = (
        torch.from_numpy(train_X).float(),
        torch.from_numpy(train_Y).float(),
        torch.from_numpy(valid_X).float(),
        torch.from_numpy(valid_Y).float(),
    )

    # 再通过Dataloader自动生成可训练的batch数据
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 损失函数
    criterion = torch.nn.MSELoss()

    valid_loss_min = float("inf")
    bad_epoch = 0

    for epoch in range(config.epoch):
        model.train()  # 切换为训练模式
        train_loss_array = []  # 记录每个batch的loss
        hidden_train = None
        for train_x, train_y in train_loader:
            train_x, train_y = train_x.to(config.device), train_y.to(config.device)
            optimizer.zero_grad()  # 训练前将梯度信息置为 0
            pred_y = model(train_x, hidden_train)  # 前向计算

            loss = criterion(pred_y, train_y)  # 计算loss
            loss.backward()  # loss反向传播
            optimizer.step()  # 更新参数

            train_loss_array.append(loss.item())

        # 以下为早停机制, 当模型连续训练config.patience个epoch都没有使验证集预测效果提升时, 就停止, 防止过拟合
        model.eval()  # 切换为评估模式
        valid_loss_array = []  # 记录每个batch的loss
        hidden_valid = None
        for valid_x, valid_y in valid_loader:
            valid_x, valid_y = valid_x.to(config.device), valid_y.to(config.device)
            pred_y = model(valid_x, hidden_valid)  # 前向计算

            loss = criterion(pred_y, valid_y)  # 计算loss

            valid_loss_array.append(loss.item())

        # 计算本轮的训练总损失和验证总损失
        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)

        # 打印损失
        print(f"Epoch {epoch}/{config.epoch}: Train Loss is {train_loss_cur:.6f}, Valid Loss is {valid_loss_cur:.6f}")

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path)  # 保存模型
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:  # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                print(" The training stops early in epoch {}".format(epoch))
                break

    return model


def predict(config, test_X, model_trained: LSTMNet = None):
    if model_trained is None:
        model_trained = LSTMNet(config).to(config.device)  # 模型实例化
        state_dict = torch.load(config.model_load_path)  # 加载训练好的模型参数
        model_trained.load_state_dict(state_dict)

    # 首先转为Tensor
    test_X = torch.from_numpy(test_X).float()
    # 再通过Dataloader自动生成可训练的batch数据
    test_loader = DataLoader(TensorDataset(test_X), batch_size=1)

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(config.device)

    # 预测过程
    model_trained.eval()  # 切换为评估模式
    hidden_predict = None
    for test_x in test_loader:
        test_x = test_x[0].to(config.device)
        pred_y = model_trained(test_x, hidden_predict)
        cur_pred = torch.squeeze(pred_y, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()  # 先去梯度信息, 如果在gpu要转到cpu, 最后要返回numpy数据
