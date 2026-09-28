import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,embed_dim):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.d_k = embed_dim // num_heads

        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):

        # Query, Key, Value shape: (batch_size, seq_length, embed_dim)
        batch_size = query.size(0)

        #Projected shape: (batch_size, num_heads, seq_length, d_k)
        Q = self.query_linear(query).reshape(batch_size,-1,self.num_heads,self.d_k).transpose(1, 2)
        K = self.key_linear(key).reshape(batch_size,-1,self.num_heads,self.d_k).transpose(1, 2)
        V = self.value_linear(value).reshape(batch_size,-1,self.num_heads,self.d_k).transpose(1, 2)
        # print(Q.shape)  # Debugging line to check the shape of Q

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = nn.functional.softmax(attn, dim=-1)
        attn_output = torch.matmul(attn, V)
        # Reshape back to (batch_size, seq_length, embed_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        attn_output = self.output_linear(attn_output)
        return attn_output

# query = torch.rand(2, 5, 128)  # Example query tensor
# key = torch.rand(2, 10, 128)    # Example

# attention = MultiHeadAttention(num_heads=4, embed_dim=128)
# output = attention(query, key, key)  # Using key as value for simplicity