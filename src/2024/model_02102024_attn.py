"""
Attention MLP (built mostly from scratch) designed for ordinal prediction tasks, such as ranking scenarios—e.g., determining that a runner placed 3rd in a race. Useful in cases where you do not want to deal with feature engineering part and vectorize it.

"""

import torch
import torch.nn as nn
import torch.optim as optim

class Ranker(nn.Module):
    def __init__(self, num_ranks, emb_dim, num_heads, dropout_rate = 0.6, seq_len = 25):
        super(Ranker, self).__init__()
        self.num_ranks = num_ranks
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.seq_len = seq_len

        # batch_first means that input and output follows (1, seq_length, emb_dim)
        self.attn = nn.MultiheadAttention(self.emb_dim, self.num_heads, dropout=self.dropout_rate, batch_first=True, bias=False)
        self.fn = nn.Linear(self.emb_dim, self.num_ranks, bias=True)

    def forward(self, Q, K, V):
        assert Q.size(1) == self.seq_len and Q.size(2) == self.emb_dim
        assert Q.shape == K.shape == V.shape

        batch_size = Q.size(0)
        attn_output, attn_weights = self.attn(Q, K, V)
        assert attn_output.size(0) == batch_size and attn_output.size(1) == self.seq_len and attn_output.size(2) == self.emb_dim, f"Expected Attn output shape to be (batch_size {self.seq_len}, {self.emb_dim}) but received {attn_output.shape}"
        assert attn_weights.size(0) == batch_size and attn_weights.size(1) == attn_weights.size(2) == self.seq_len, f"Expected Attn output weight shape to be (batch_size {self.seq_len}, {self.seq_len}) but received {attn_weights.shape}"

        output = attn_output.softmax(-1)
        x = self.fn(output)
        assert x.dim() == 3 and x.size(1) == self.seq_len and x.size(2) == self.num_ranks
        x = x.sum(axis=1).softmax(-1).unsqueeze(1)
        assert x.dim() == 3 and x.size(0) == batch_size and x.size(2) == self.num_ranks

        # print(f"Prediction:\n", torch.round(x * 1e2) / 1e2)
        return x


def display_results(model_output, target, threshold: float = 0.3):
    """ counts by batch input higher than threshold """
    assert model_output.dim() == 3
    results = (model_output > threshold).sum(axis=0)
    target_results = (target > threshold).sum(axis=0)
    print("Model predicted ranks: ", results, "Number of correct", (results == target_results).sum(axis=1).item())

def mock_dataset(batch_size, emb_dim, seq_len):
    q = torch.randn(batch_size, seq_len, emb_dim)
    k = torch.randn(batch_size, seq_len, emb_dim)
    v = torch.randn(batch_size, seq_len, emb_dim)
    target = torch.tensor([
        [[1, 0, 0]],
        [[1, 1, 0]],
        [[1, 1, 1]]
    ], dtype=torch.float)
    print("Q:\n", q.shape)
    print("K:\n", k.shape)
    print("V:\n", v.shape)
    print("Target:\n", target.shape)
    print('--'* 10)
    return q, k, v, target

def trial(max_epoch: int = 25):
    emb_dim = 4
    num_heads = 2
    dropout_rate = 0.6
    num_ranks = 3
    seq_len = 25
    batch_size = 3

    q, k, v, target = mock_dataset(batch_size, emb_dim, seq_len)
    model = Ranker(num_ranks, emb_dim, num_heads, dropout_rate, seq_len)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(int(max_epoch)):
        optimizer.zero_grad()

        # Assuming model(q, k, v) returns predictions of shape (batch_size, seq_len, 1)
        output = model(q, k, v)
        loss = criterion(output, target)
        display_results(output, target)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{max_epoch} - Loss: {loss.item():.4f}")

trial()