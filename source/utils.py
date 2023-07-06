import torch.nn as nn


class GlobalGatedAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()

        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout)
        )

        self.attention_c = nn.Linear(*[hidden_dim, output_dim])
        # self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)

        A = self.attention_c(A)

        return A
        # attention = self.softmax(A)

        # return attention
