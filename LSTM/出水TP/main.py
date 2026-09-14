from sklearn.model_selection import train_test_split
from model.model_pytorch import train, predict
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


class Config:
    def __init__(self):
        self.feature_columns = list(range(1, 24))  # 要作为feature的列, 按原数据从0开始计算，也可以用list 如[2,4,6,8]设置
        self.label_columns = [23]  # 要预测的列, 按原数据从0开始计算, 如同时预测第4列和第5列可写为 [4,5]
        self.label_in_feature_index = (lambda x, y: [x.index(i) for i in y])(self.feature_columns, self.label_columns)  # 因为feature不一定从0开始

        # 网络参数
        self.input_size = len(self.feature_columns)
        self.ouput_size = len(self.label_columns)

        self.lstm_hidden_size = 16  # GRU的隐藏层维度
        self.lstm_layers = 2  # GRU的堆叠层数
        self.dropout_rate = 0.2  # dropout概率
        self.time_step = 4  # 这个参数很重要，是设置用前多少天的数据来预测，也是GRU的time step数，请保证训练数据量大于它

        # 路径参数
        self.train_data_path = "./data/input_22_7.7-23_12.13_TP.csv"

        # 训练好的模型的存储位置
        self.model_save_path = "./trained_model/model_lstm.pth"

        # 要调用的训练好的模型的存储位置
        self.model_load_path = "./trained_model/model_lstm.pth"

        # 存放预测和测试的结果数据csv
        self.csv_file_path = "./csv/result.csv"

        # 最后预测效果图保存路径
        self.predict_effect_path = "./fig/predict.svg"

        # 训练集, 测试集和验证集的划分
        self.train_data_rate = 0.95  # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
        self.valid_data_rate = 0.15  # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

        # 训练参数
        self.batch_size = 64
        self.learning_rate = 0.001
        self.epoch = 400  # 整个训练集被训练多少遍，不考虑早停的前提下
        self.patience = 400  # 训练多少epoch，验证集没提升就停掉
        self.random_seed = 42  # 随机种子，保证可复现
        self.shuffle_train_data = True  # 是否对训练数据做shuffle

        # 训练和预测的启停
        self.do_train = True
        self.do_predict = True
        self.use_trained_model = True

        # GRU和全连接层的连接关系选择
        self.op = 2

        # 决定是否使用cuda
        self.use_cuda = True
        self.device = torch.device("cuda:0" if self.use_cuda and torch.cuda.is_available() else "cpu")  # CPU训练还是GPU


class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()  # 从csv中读取数据

        self.data_num = self.data.shape[0]  # 总共的数据量
        self.train_num = int(self.data_num * self.config.train_data_rate)  # 训练集的数据量
        self.test_num = self.data_num - self.train_num  # 测试集的数据量

        self.scalerX = preprocessing.MinMaxScaler()  # 用于归一化输入特征
        self.scalerY = preprocessing.MinMaxScaler()  # 用于归一化输出

    def read_data(self):
        init_data = pd.read_csv(self.config.train_data_path, usecols=self.config.feature_columns)
        return init_data.values, init_data.columns.tolist()  # columns.tolist()是获取列名

    def get_train_and_valid_data(self):
        # 获取训练集
        feature_data = self.data[: self.train_num]
        # 将延后time_step行的数据作为label
        label_data = self.data[self.config.time_step : self.config.time_step + self.train_num, self.config.label_in_feature_index]

        # 每time_step行数据会作为一个样本, 两个样本错开一行, 比如: 1-20行, 2-21行
        train_X = [feature_data[i : i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
        train_Y = [label_data[i] for i in range(self.train_num - self.config.time_step)]

        # 从训练集中分离出验证集, 并打乱
        train_X, valid_X, train_Y, valid_Y = train_test_split(
            train_X,
            train_Y,
            test_size=self.config.valid_data_rate,
            random_state=self.config.random_seed,
            shuffle=self.config.shuffle_train_data,
        )

        # 转换为ndarray
        train_X, valid_X, train_Y, valid_Y = np.array(train_X), np.array(valid_X), np.array(train_Y), np.array(valid_Y)

        # 由于标准化需要的维度必须小于2, 所以对于输入先采用reshape
        train_X, valid_X = train_X.reshape((train_X.shape[0], -1)), valid_X.reshape((valid_X.shape[0], -1))

        # 将训练集标准化,
        norm_train_X = self.scalerX.fit_transform(train_X)
        norm_train_Y = self.scalerY.fit_transform(train_Y)

        # 再将验证集标准化
        norm_valid_X = self.scalerX.transform(valid_X)
        norm_valid_Y = self.scalerY.transform(valid_Y)

        # 之后再reshape回去
        norm_train_X = norm_train_X.reshape((norm_train_X.shape[0], -1, self.config.input_size))
        norm_valid_X = norm_valid_X.reshape((norm_valid_X.shape[0], -1, self.config.input_size))

        return norm_train_X, norm_train_Y, norm_valid_X, norm_valid_Y

    def get_test_data(self, return_label_data=False):
        # 获取测试集
        feature_data = self.data[self.train_num :]
        # 将延后time_step行的数据作为label
        label_data = self.data[self.config.time_step + self.train_num :, self.config.label_in_feature_index]

        # 每time_step行数据会作为一个样本, 两个样本错开一行, 比如: 1-20行, 2-21行
        test_X = [feature_data[i : i + self.config.time_step] for i in range(self.test_num - self.config.time_step)]
        test_Y = [label_data[i] for i in range(self.test_num - self.config.time_step)]

        # 转换为ndarray
        test_X, test_Y = np.array(test_X), np.array(test_Y)

        # 由于标准化需要的维度必须小于2, 所以对于输入先采用reshape
        test_X = test_X.reshape((test_X.shape[0], -1))

        # 将测试集标准化
        norm_test_X = self.scalerX.transform(test_X)
        norm_test_Y = self.scalerY.transform(test_Y)

        # 之后再reshape回去
        norm_test_X = norm_test_X.reshape((norm_test_X.shape[0], -1, self.config.input_size))

        if return_label_data:  # 实际应用中的测试集是没有label数据的
            return norm_test_X, norm_test_Y
        return norm_test_X


"""绘图"""


def plot(config, test_Y, pred_Y_mean):
    f, ax = plt.subplots(1, 1)
    x_axis = np.arange(test_Y.shape[0])
    ax.plot(x_axis, test_Y, label="real value", c="r")  # 绘制真实值为红色的line
    ax.plot(x_axis, pred_Y_mean, label="pred value", c="b")  # 绘制预测的平均值为蓝色的line
    ax.grid(linestyle="--", alpha=0.5)  # 绘制格栅
    ax.legend()  # 绘制标签
    # 标题
    ax.set_title("Effluent TP")
    # 坐标轴
    ax.set_xlabel("2 hours/sample")
    ax.set_ylabel("mg/L")
    plt.savefig(config.predict_effect_path)


"""存储到csv文件中"""


def save_to_csv(config, test_Y, pred_Y_mean):
    task = config.ouput_size
    for i in range(task):
        data = {
            "test_Y": test_Y[:, i].tolist(),
            "pred_Y": pred_Y_mean[:, i].tolist(),
        }
        df = pd.DataFrame(data)
        df.to_csv(config.csv_file_path, index=False)


"""计算拟合指标"""


def measurement(test_Y, pred_Y_mean):
    mae = np.mean(np.abs(pred_Y_mean - test_Y))
    rmse = np.mean(np.power(pred_Y_mean - test_Y, 2))
    mape = np.mean(np.abs((pred_Y_mean - test_Y) / test_Y)) * 100
    r2 = 1 - np.sum((test_Y - pred_Y_mean) ** 2) / np.sum((test_Y - np.mean(test_Y)) ** 2)
    print(f"模型回归评价指标:\nMAE:{mae:.6f}\nRMSE:{rmse:.6f}\nMAPE:{mape:.6f}%\nR2:{r2:.6f}")


def main(config: Config):
    np.random.seed(config.random_seed)  # 设置随机数种子, 保证可复现
    data_gainer = Data(config)

    train_X, train_Y, valid_X, valid_Y = data_gainer.get_train_and_valid_data()
    test_X, test_Y = data_gainer.get_test_data(return_label_data=True)

    if config.do_train:  # 如果开启训练
        model = train(config, [train_X, train_Y, valid_X, valid_Y])

    if config.do_predict:  # 如果开启测试
        if config.use_trained_model:  # 如果使用训练好的模型
            model = None  # model输入置空
        pred_Y = predict(config, test_X, model)

    # 将结果反标准化
    test_Y = data_gainer.scalerY.inverse_transform(test_Y.reshape(-1, 1))
    pred_Y = data_gainer.scalerY.inverse_transform(pred_Y.reshape(-1, 1))

    # 绘制图像
    plot(config, test_Y, pred_Y)

    # 计算指标
    measurement(test_Y, pred_Y)

    # 保存数据到csv
    save_to_csv(config, test_Y, pred_Y)


if __name__ == "__main__":
    # 主执行入口
    con = Config()
    main(con)
