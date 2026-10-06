import torch
import torch.nn as nn
from .multihead_attention import MultiHeadAttention

class FeedForwardNetwork(nn.Module):
    def __init__(self,input_dim,hidden_dim,dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self,x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self,embed_dim,num_heads,ffn_dim,dropout):
        super().__init__()

        self.self_attention = MultiHeadAttention(num_heads, embed_dim)
        
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout) # Keeping same dropout for FFN as for encoder for now

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x):
        # Multi-head attention with residual connection
        x = self.norm1(x + self.dropout1(self.self_attention(x, x, x)))

        # Feed-forward network with residual connection
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x

class TransformerEncoder(nn.Module):
    def __init__(self,num_layers,embed_dim,num_heads,ffn_dim,dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x


# # Example usage
# if __name__ == "__main__":
#     # Example input: batch_size=2, seq_length=5, embed_dim=128
#     x = torch.rand(2, 5, 128)

#     encoder = TransformerEncoder(num_layers=2, embed_dim=128, num_heads=4, ffn_dim=512, dropout=0.1)
#     output = encoder(x)
#     print(output.shape)  # Should be (2, 5, 128)
