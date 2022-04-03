import torch.nn as nn
from typing import List, Optional


class LSTMModel(nn.Module):
    def __init__(self, inputDim: int, hiddenDim: int, outputDim: int) -> None:
        super(LSTMModel, self).__init__()

        self.rnn = nn.LSTM(
            input_size=inputDim, hidden_size=hiddenDim, batch_first=True
        )  # type:ignore
        self.output_layer = nn.Linear(hiddenDim, outputDim)

    def forward(
        self, inputs: List[float], hidden0: Optional[float] = None
    ) -> float:
        output, (hidden, cell) = self.rnn(inputs, hidden0)  # LSTM層
        output = self.output_layer(output[:, -1, :])  # 出力層(=全結合層)

        return output  # type:ignore
