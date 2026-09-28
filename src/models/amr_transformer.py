import torch
import torch.nn as nn
# append src to sys.path to import MultiHeadAttention
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'transformer'))
from transformer.transformer_encoder import TransformerEncoder
from transformer.transformer_decoder import TransformerDecoder
from transformer.positional_encoding import PositionalEncoding


class amrTransformer(nn.Module):
    def __init__(self, embed_dim = 128, num_waypoints=5):
        super().__init__()

        # Initialize the encoder and decoder and positional encoding
        self.encoder = TransformerEncoder(num_layers=6,embed_dim=embed_dim,num_heads=8, ffn_dim=1024,dropout=0.1)
        self.decoder = TransformerDecoder(num_layers=6,embed_dim=embed_dim,num_heads=8, ffn_dim=1024,dropout=0.1)

        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=5000)

        # Using nn.Embedding for learnable action query embeddings
        self.action_query_embeddings = nn.Embedding(num_waypoints, embed_dim)


        # Linear layer to get the final output from the decoder
        self.output_linear = nn.Linear(embed_dim, 2)  # Assuming output is 2D coordinates (x, y)

    def forward(self, observations, goals):

        # Concat obs and goals
        encoder_input = torch.cat((observations,goals), dim = 1)

        # Apply positional encoding to the encoder input
        encoder_input = self.positional_encoding(encoder_input)

        # Pass input to the encoder
        encoder_memory = self.encoder(encoder_input)

        # Prepare action queries for the decoder
        batch_size = observations.size(0)
        action_queries = self.action_query_embeddings.weight # access the weights
        action_queries = action_queries.unsqueeze(0) # Add batch dimension to make it (1, num_waypoints, embed_dim)
        action_queries = action_queries.expand(batch_size, -1, -1)  # Expand to (batch_size, num_waypoints, embed_dim)

        # Pass action queries and encoder memory to the decoder
        decoder_output = self.decoder(action_queries, encoder_memory)

        # Get the final output from the decoder
        output = self.output_linear(decoder_output)
        return output

model = amrTransformer(embed_dim=128, num_waypoints=5)

observations = torch.randn(2, 4, 128)
goals = torch.randn(2, 1, 128)

waypoints = model(observations, goals)

assert waypoints.shape == (2, 5, 2)

loss = waypoints.square().mean()
loss.backward()

print("Output shape:", waypoints.shape)
print("Action query gradient:", model.action_query_embeddings.weight.grad is not None)

for name, param in model.named_parameters():
    if param.grad is None:
        print("Missing gradient:", name)

assert torch.isfinite(waypoints).all()

for name, param in model.named_parameters():
    if param.grad is not None:
        assert torch.isfinite(param.grad).all(), name