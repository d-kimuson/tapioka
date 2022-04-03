import numpy as np
import torch
import math
import torch.nn as nn
from torch import Tensor
from os import environ
from typing import List, Tuple
from torch.optim import SGD
from pydantic import BaseModel

from projects.sin.models.lstm import LSTMModel

DataType = List[List[List[float]]]
TestType = List[List[float]]


class DataSet(BaseModel):
    data: DataType
    test: TestType


class LSTMRunner:
    lowest_loss_epoch: int = -1
    lowest_loss: float = float("inf")

    def __init__(self) -> None:
        print("Hello")

    def run(self) -> None:
        self.__fix_seed(0)

        batch_size = 10
        epochs_num = 100  # traningのepoch回数

        phase_data_dict = {
            "train": self.__make_data_set(100),
            "test": self.__make_data_set(1000),
        }

        model = LSTMModel(1, 5, 1)
        loss_func = nn.MSELoss()  # 評価関数の宣言
        optimizer = SGD(model.parameters(), lr=0.01)  # 最適化関数の宣言

        for epoch_count in range(epochs_num):
            training_accuracy = 0.0
            test_accuracy = 0.0

            for phase in ["train", "test"]:
                phase_data = phase_data_dict[phase]
                data_size = len(phase_data.data)
                epoch_loss = 0.0
                accuracy = 0.0

                if phase == "train":
                    model.train()
                else:
                    model.eval()

                # batch
                for i in range(int(data_size / batch_size)):

                    optimizer.zero_grad()

                    inputs, outputs = self.__make_random_batch(
                        phase_data.data, phase_data.test, batch_size
                    )

                    y_pred = model(inputs)
                    y_pred.to(torch.float64)

                    outputs.to(torch.float64)
                    loss = loss_func(y_pred, outputs)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.data.item() * batch_size
                    accuracy += np.sum(
                        np.abs((y_pred.data - outputs.data).numpy()) < 0.1
                    )  # outputとlabelの誤差が0.1以内なら正しいとみなす。

                if phase == "train":
                    training_accuracy = accuracy / data_size
                else:
                    test_accuracy = accuracy / data_size
                    self.__update_min_loss(epoch_loss, epoch_count)

                    print(
                        "%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f"
                        % (
                            epoch_count + 1,
                            epoch_loss,
                            training_accuracy,
                            test_accuracy,
                        )
                    )

    def __fix_seed(self, seed: int) -> None:
        environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __update_min_loss(self, loss: float, epoch: int) -> None:
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            self.lowest_loss_epoch = epoch

    def __make_random_batch(
        self,
        train_x: DataType,
        train_t: TestType,
        batch_size: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        """
        train_x, train_tを受け取ってbatch_x, batch_tを返す。
        """
        batch_x: DataType = []
        batch_t: TestType = []

        for _ in range(batch_size):
            idx = np.random.randint(0, len(train_x) - 1)
            batch_x.append(train_x[idx])
            batch_t.append(train_t[idx])

        return torch.tensor(batch_x), torch.tensor(batch_t)

    def __make_data_set(
        self,
        data_size: int,
        data_length: int = 50,
        freq: float = 60.0,
        noise: float = 0.02,
    ) -> DataSet:
        """
        params
        data_size : データセットサイズ
        data_length : 各データの時系列長
        freq : 周波数
        noise : ノイズの振幅
        returns
        train_x : トレーニングデータ（t=1,2,...,size-1の値)
        train_t : トレーニングデータのラベル（t=sizeの値）
        """
        train_x: List[List[List[float]]] = []
        train_t: List[List[float]] = []

        for offset in range(data_size):
            train_x.append(
                [
                    [
                        math.sin(2 * math.pi * (offset + i) / freq)
                        + np.random.normal(loc=0.0, scale=noise)
                    ]
                    for i in range(data_length)
                ]
            )
            train_t.append(
                [math.sin(2 * math.pi * (offset + data_length) / freq)]
            )

        return DataSet(**{"data": train_x, "test": train_t})
