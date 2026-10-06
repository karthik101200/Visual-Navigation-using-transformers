import torch
import torch.nn as nn
# append src to sys.path to import MultiHeadAttention
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'transformer'))
from transformer.transformer_encoder import TransformerEncoder
from transformer.transformer_decoder import TransformerDecoder
from transformer.positional_encoding import PositionalEncoding
from cnn_encoder import ImageEncoder

class amrTransformer(nn.Module):
    def __init__(self, embed_dim = 128, num_waypoints=5):
        super().__init__()

        # Image encoder to process observations and goals
        self.observation_encoder = ImageEncoder(input_channels=3, output_dim=embed_dim)
        self.goal_encoder = ImageEncoder(input_channels=6, output_dim=embed_dim)

        # Initialize the encoder and decoder and positional encoding
        self.encoder = TransformerEncoder(num_layers=6,embed_dim=embed_dim,num_heads=8, ffn_dim=1024,dropout=0.1)
        self.decoder = TransformerDecoder(num_layers=6,embed_dim=embed_dim,num_heads=8, ffn_dim=1024,dropout=0.1)

        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=5000)

        # Using nn.Embedding for learnable action query embeddings
        self.action_query_embeddings = nn.Embedding(num_waypoints, embed_dim)


        # Linear layer to get the final output from the decoder
        self.output_linear = nn.Linear(embed_dim, 2)  # Assuming output is 2D coordinates (x, y)

    def forward(self, observations, goals):

        # Encode Observation using Image Encoder
        B,N,C,H,W = observations.shape  # B: batch size, N: number of observations, C: channels, H: height, W: width

        latest_observation = observations[:, -1, :, :, :]  # Get the latest observation for each batch
        latest_observation = latest_observation.view(B, C, H, W)

        observations = observations.view(-1, C, H, W)  # Reshape to (B*N, C, H, W)
        observations = self.observation_encoder(observations)  # Now shape is (B*N, embed_dim)
        observations = observations.view(B, N, observations.shape[-1])  # Reshape back to (B, N, embed_dim)

        # Encode Goal by concatenating latest observation and goal image, then passing through Image Encoder
        B,G,C,H,W = goals.shape  # B: batch size, G: number of goals (should be 1), C: channels, H: height, W: width

        goal = goals[:, 0, :, :, :]  # Extract the single goal image
        goal_input = torch.cat((latest_observation, goal), dim=1)  # Concatenate along channel dimension to get (B, 6, H, W)
        goal = self.goal_encoder(goal_input)  # Now shape is (B, embed_dim)
        goal = goal.view(B, 1, goal.shape[-1])  # Reshape to (B, 1, embed_dim)


        # Concat obs and goals
        encoder_input = torch.cat((observations, goal), dim = 1)

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

observations = torch.randn(2, 4, 3, 64, 64)  # B, N, C, H, W
goal = torch.randn(2, 1, 3, 64, 64)  # B, N, C, H, W

waypoints = model(observations, goal)

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