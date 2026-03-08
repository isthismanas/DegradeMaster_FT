import torch
import torch.nn as nn

from model import EGNNConv, GATTConv, GraphConv


class CrossAttentionGNNRegressor(nn.Module):
    """Fuses pocket/ligand embeddings with self-attention before regression."""

    def __init__(
        self,
        protac_encoder: nn.Module,
        ligase_encoder: nn.Module,
        target_encoder: nn.Module,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.protac_encoder = protac_encoder
        self.ligase_encoder = ligase_encoder
        self.target_encoder = target_encoder

        self.token_pos = nn.Parameter(torch.zeros(1, 3, hidden_size))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_size, 2)

    def forward(self, protac_graphs, ligase_pocket, target_pocket, feature):
        fea_protac = feature[:, :167]
        fea_tar = feature[:, 167:197]
        fea_e3 = feature[:, 197:227]

        v_protac = self.protac_encoder(protac_graphs, fea_protac)
        v_ligase = self.ligase_encoder(ligase_pocket, fea_e3)
        v_target = self.target_encoder(target_pocket, fea_tar)

        tokens = torch.stack((v_protac, v_ligase, v_target), dim=1)
        tokens = tokens + self.token_pos
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = self.norm(attn_out + tokens)

        flattened = attn_out.reshape(attn_out.shape[0], -1)
        latent = self.mlp(flattened)
        preds = self.out(latent)
        return preds, latent


def build_cross_attention_gnn(args):
    if args.conv_name == 'GCN':
        ligase_encoder = GraphConv(num_embeddings=118, graph_dim=args.e3_dim, hidden_size=args.hidden_size)
        target_encoder = GraphConv(num_embeddings=118, graph_dim=args.tar_dim, hidden_size=args.hidden_size)
        protac_encoder = GraphConv(num_embeddings=118, graph_dim=args.protac_dim, hidden_size=args.hidden_size)
    elif args.conv_name == 'GAT':
        ligase_encoder = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)
        target_encoder = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)
        protac_encoder = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)
    elif args.conv_name == 'EGNN':
        ligase_encoder = EGNNConv(
            num_embeddings=118,
            in_node_nf=1,
            in_edge_nf=1,
            graph_nf=args.e3_dim,
            hidden_nf=args.hidden_size,
            n_layers=args.n_layers,
            node_attr=0,
            attention=args.attention,
        )
        protac_encoder = EGNNConv(
            num_embeddings=118,
            in_node_nf=1,
            in_edge_nf=1,
            graph_nf=args.protac_dim,
            hidden_nf=args.hidden_size,
            n_layers=args.n_layers,
            node_attr=0,
            attention=args.attention,
        )
        target_encoder = EGNNConv(
            num_embeddings=118,
            in_node_nf=1,
            in_edge_nf=1,
            graph_nf=args.tar_dim,
            hidden_nf=args.hidden_size,
            n_layers=args.n_layers,
            node_attr=0,
            attention=args.attention,
        )
    else:
        raise ValueError(f'Unsupported conv_name: {args.conv_name}')

    dropout = getattr(args, 'dropout', 0.1)
    num_heads = getattr(args, 'num_attn_heads', 4)
    return CrossAttentionGNNRegressor(
        protac_encoder=protac_encoder,
        ligase_encoder=ligase_encoder,
        target_encoder=target_encoder,
        hidden_size=args.hidden_size,
        num_heads=num_heads,
        dropout=dropout,
    )
