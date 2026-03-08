import torch
import torch.nn as nn

from model import EGNNConv, GATTConv, GraphConv


class PDC50BoundedGNNRegressor(nn.Module):
    """
    Predicts pDC50 (unbounded linear head) and bounded Dmax (sigmoid scaled).

    The intended target format is:
    - column 0: pDC50 = -log10(DC50 in molar)
    - column 1: Dmax in percent [0, dmax_upper]
    """

    def __init__(
        self,
        protac_encoder: nn.Module,
        ligase_encoder: nn.Module,
        target_encoder: nn.Module,
        hidden_size: int,
        dropout: float = 0.1,
        dmax_upper: float = 100.0,
    ) -> None:
        super().__init__()
        self.protac_encoder = protac_encoder
        self.ligase_encoder = ligase_encoder
        self.target_encoder = target_encoder
        self.dmax_upper = float(dmax_upper)

        self.shared = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
        )

        self.pdc50_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.dmax_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, protac_graphs, ligase_pocket, target_pocket, feature):
        fea_protac = feature[:, :167]
        fea_tar = feature[:, 167:197]
        fea_e3 = feature[:, 197:227]

        v_protac = self.protac_encoder(protac_graphs, fea_protac)
        v_ligase = self.ligase_encoder(ligase_pocket, fea_e3)
        v_target = self.target_encoder(target_pocket, fea_tar)

        fused = torch.cat((v_protac, v_ligase, v_target), dim=1)
        latent = self.shared(fused)

        pdc50_pred = self.pdc50_head(latent)
        dmax_pred = torch.sigmoid(self.dmax_head(latent)) * self.dmax_upper
        preds = torch.cat((pdc50_pred, dmax_pred), dim=1)
        return preds, latent


def build_pdc50_bounded_gnn(args):
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
    dmax_upper = getattr(args, 'dmax_upper', 100.0)
    return PDC50BoundedGNNRegressor(
        protac_encoder=protac_encoder,
        ligase_encoder=ligase_encoder,
        target_encoder=target_encoder,
        hidden_size=args.hidden_size,
        dropout=dropout,
        dmax_upper=dmax_upper,
    )
