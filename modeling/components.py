import torch
from torch import nn

class ConcatHiddenStates(nn.Module):
    """
    Concatenates the last `num_concat` hidden states.
    """

    def __init__(self, num_concat) -> None:
        super().__init__()

        self.num_concat = num_concat

    def forward(self, all_hidden_states, **kwargs):
        return torch.cat([hs for hs in all_hidden_states[-self.num_concat :]], dim=-1)

class BiLSTMHead(nn.Module):
    """
    WIP
    """

    def __init__(self, embedding_dim, hidden_dim) -> None:
        super().__init__()

        self.bilstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=1, bidirectional=True
        )

    def forward(self, x):
        x, _ = self.bilstm(x)
        return x


class CLSHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states, **kwargs):
        return hidden_states[:, 0, :]


class MeanPoolHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states, attention_mask, **kwargs):
        x = hidden_states
        mask = attention_mask.unsqueeze(-1).expand(x.size())
        x = torch.sum(x * mask, 1)
        x = x / (mask.sum(1) + 1e-8)
        return x


class MaxPoolHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states, **kwargs):
        max_pooled, _ = torch.max(hidden_states, 1)
        return max_pooled


class MeanMaxPoolHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max = MaxPoolHead()
        self.mean = MeanPoolHead()

    def forward(self, hidden_states, **kwargs):
        max_pooled = self.max(hidden_states)
        mean_pooled = self.mean(hidden_states, **kwargs)

        return torch.cat([max_pooled, mean_pooled], dim=1)

class MultiSampleDropout(nn.Module):
    def __init__(self, dropout_probs) -> None:
        super().__init__()

        self.dropouts = [nn.Dropout(p=p) for p in dropout_probs]

    def forward(self, hidden_states, linear, labels, loss_fn, layer_nm):
        # if not using output layer_nm, pass nn.Identity()

        logits = [linear(layer_nm(d(hidden_states))) for d in self.dropouts]

        losses = [loss_fn(log.view(-1, labels.size(1)), labels) for log in logits]

        logits = torch.mean(torch.stack(logits, dim=0), dim=0)
        loss = torch.mean(torch.stack(losses, dim=0), dim=0)

        return (loss, logits)
        
class GatedDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.hidden_size, 512, bias=False)
        self.wi_1 = nn.Linear(config.hidden_size, 512, bias=False)
        self.wo = nn.Linear(512, config.num_labels, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act_fn = ACT2FN[config.gated_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act_fn(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states