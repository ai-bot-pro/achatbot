import torch
from torch import nn
import torch.nn.functional as F
import torchaudio


class Wav2Vec2ForEndpointingTorch(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        # Load Wav2Vec2 model from torchaudio
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2 = bundle.get_model()

        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Initialize weights for classifier and attention pooling layers
        self._init_weights(self.classifier)
        self._init_weights(self.pool_attention)

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def attention_pool(self, hidden_states, attention_mask):
        attention_weights = self.pool_attention(hidden_states)

        if attention_mask is None:
            raise ValueError("attention_mask must be provided for attention pooling")

        attention_weights = attention_weights + (
            (1.0 - attention_mask.unsqueeze(-1).to(attention_weights.dtype)) * -1e9
        )

        attention_weights = F.softmax(attention_weights, dim=1)
        weighted_sum = torch.sum(hidden_states * attention_weights, dim=1)
        return weighted_sum

    def forward(self, input_values, attention_mask=None, labels=None):
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=-1)
            hidden_states, _ = self.wav2vec2(input_values, lengths=lengths)
        else:
            hidden_states, _ = self.wav2vec2(input_values)

        if attention_mask is not None:
            input_length = attention_mask.size(1)
            hidden_length = hidden_states.size(1)
            if hidden_length > 0:
                ratio = input_length / hidden_length
                indices = (torch.arange(hidden_length, device=attention_mask.device) * ratio).long()
                indices = torch.clamp(indices, 0, input_length - 1)
                pooled_attention_mask = attention_mask[:, indices]
            else:
                pooled_attention_mask = torch.zeros(
                    hidden_states.shape[0], 0, device=hidden_states.device, dtype=torch.bool
                )
        else:
            pooled_attention_mask = torch.ones(
                hidden_states.shape[0], hidden_states.shape[1], device=hidden_states.device
            ).bool()

        pooled = self.attention_pool(hidden_states, pooled_attention_mask)
        logits = self.classifier(pooled)

        if torch.isnan(logits).any():
            raise ValueError("NaN values detected in logits")

        if labels is not None:
            pos_weight = (
                ((labels == 0).sum() / (labels == 1).sum()).clamp(min=0.1, max=10.0)
                if (labels == 1).sum() > 0
                else torch.tensor(1.0)
            )
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

            l2_lambda = 0.01
            l2_reg = torch.tensor(0.0, device=logits.device)
            for param in self.classifier.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            return {"loss": loss, "logits": logits}

        return {"logits": logits}
