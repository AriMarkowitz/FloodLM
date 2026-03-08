"""
Model_3: Non-autoregressive encoder-decoder heterogeneous GNN.

Architecture:
  Encoder: HeteroTransportCell (same GRU + message passing as Model_2) × history_len steps
           → h_enc[nt]: [N_nt, h_dim]  (rich per-node hidden state at t=history_len)

  Decoder: Parallel MLP applied independently to each future timestep:
           input_t[node] = cat(h_enc[node], rain_future_t[node], sin/cos(t/T_max))
           output_t = Decoder_MLP(input_t)
           All T_future timesteps computed in one batched MLP call — no BPTT through horizon.

GNN cell code copied from src/model.py (no imports from parent src/).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import MessagePassing, HeteroConv, GATv2Conv as _GATv2Conv


# ============================================================
# Utilities
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# ============================================================
# Message passing modules (copied from src/model.py)
# ============================================================

class StaticDynamicEdgeMP(MessagePassing):
    """
    Static-gate + dynamic-gate message passing.
    Used for homogeneous edges and cross-type edges with rich features.
    """
    def __init__(self, h_dim, node_static_dim_src, node_static_dim_dst,
                 edge_static_dim, msg_dim, hidden_dim, dropout=0.0, aggr="add"):
        super().__init__(aggr=aggr)
        self.h_dim = h_dim
        self.msg_dim = msg_dim
        self.edge_static_embed = MLP(
            in_dim=edge_static_dim + node_static_dim_src + node_static_dim_dst,
            hidden_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout)
        self.base_weight = nn.Linear(hidden_dim, 1)
        self.dynamic_gate = MLP(in_dim=2 * h_dim, hidden_dim=hidden_dim, out_dim=1, dropout=dropout)
        self.payload = MLP(in_dim=h_dim, hidden_dim=hidden_dim, out_dim=msg_dim, dropout=dropout)

    def _set_context(self, h_src, h_dst, edge_attr_static, x_static_src, x_static_dst):
        self._h_src = h_src
        self._h_dst = h_dst
        self._edge_attr_static = edge_attr_static
        self._x_static_src = x_static_src
        self._x_static_dst = x_static_dst

    def forward(self, x=None, edge_index=None, **kwargs):
        return self.propagate(
            edge_index=edge_index,
            size=(self._h_src.size(0), self._h_dst.size(0)),
            h_src=self._h_src, h_dst=self._h_dst,
            edge_attr_static=self._edge_attr_static,
            x_static_src=self._x_static_src,
            x_static_dst=self._x_static_dst,
        )

    def message(self, h_src_j, h_dst_i, edge_attr_static, x_static_src_j, x_static_dst_i):
        static_cat = torch.cat([edge_attr_static, x_static_src_j, x_static_dst_i], dim=-1)
        u_e = self.edge_static_embed(static_cat)
        b_e = F.softplus(self.base_weight(u_e))
        gate_in = torch.cat([h_src_j, h_dst_i], dim=-1)
        g_e = torch.sigmoid(self.dynamic_gate(gate_in))
        v = self.payload(h_src_j)
        return (b_e * g_e) * v


class GATv2CrossTypeMP(MessagePassing):
    """GATv2-based message passing for cross-type / virtual-node edges."""
    def __init__(self, h_dim, msg_dim, hidden_dim, heads=4, dropout=0.0):
        super().__init__(aggr="add")
        self.h_dim = h_dim
        self.msg_dim = msg_dim
        self.heads = heads
        self.gatv2 = _GATv2Conv(
            in_channels=(h_dim, h_dim), out_channels=msg_dim // heads, heads=heads,
            dropout=dropout, concat=True, add_self_loops=False, residual=True)
        self.ffn = MLP(in_dim=msg_dim, hidden_dim=hidden_dim, out_dim=msg_dim, dropout=dropout)

    def _set_context(self, h_src, h_dst, **kwargs):
        self._h_src = h_src
        self._h_dst = h_dst

    def forward(self, x=None, edge_index=None, **kwargs):
        out = self.gatv2((self._h_src, self._h_dst), edge_index)
        return self.ffn(out)


# ============================================================
# Encoder GRU cell (copied from src/model.py HeteroTransportCell)
# ============================================================

class HeteroTransportCell(nn.Module):
    """Single GRU timestep over a heterogeneous graph. B=1 in Model_3."""
    def __init__(self, node_types, edge_types, node_static_dims, node_dyn_input_dims,
                 edge_static_dims, h_dim=96, msg_dim=64, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.h_dim = h_dim
        self.msg_dim = msg_dim
        self._hidden_dim = hidden_dim

        if isinstance(hidden_dim, dict):
            _hid_default = next(iter(hidden_dim.values()), 128)
            def _hid(rel): return hidden_dim.get(rel, _hid_default)
        else:
            def _hid(rel): return hidden_dim

        _ctx_types = {"ctx1d", "ctx2d", "global"}
        conv_dict = {}
        for (src, rel, dst) in edge_types:
            e_dim = edge_static_dims[(src, rel, dst)]
            if (rel in ("oneDtwoD", "twoDoneD") and e_dim == 1) or bool(_ctx_types & {src, dst}):
                mp = GATv2CrossTypeMP(h_dim=h_dim, msg_dim=msg_dim, hidden_dim=_hid(rel),
                                     heads=4, dropout=dropout)
            else:
                mp = StaticDynamicEdgeMP(
                    h_dim=h_dim, node_static_dim_src=node_static_dims[src],
                    node_static_dim_dst=node_static_dims[dst], edge_static_dim=e_dim,
                    msg_dim=msg_dim, hidden_dim=_hid(rel), dropout=dropout, aggr="add")
            conv_dict[(src, rel, dst)] = mp

        self.mp_modules = nn.ModuleDict({
            f"{src}_{rel}_{dst}": mp for (src, rel, dst), mp in conv_dict.items()})
        self.edge_types = list(conv_dict.keys())
        self.hetero_conv = HeteroConv(conv_dict, aggr="sum")

        self.dyn_proj = nn.ModuleDict({nt: nn.Linear(node_dyn_input_dims[nt], msg_dim) for nt in node_types})
        self.update = nn.ModuleDict({nt: nn.GRUCell(input_size=2*msg_dim, hidden_size=h_dim) for nt in node_types})
        self.h_norm   = nn.ModuleDict({nt: nn.LayerNorm(h_dim) for nt in node_types})
        self.msg_norm = nn.ModuleDict({nt: nn.LayerNorm(msg_dim) for nt in node_types})
        self.dyn_norm = nn.ModuleDict({nt: nn.LayerNorm(msg_dim) for nt in node_types})

    def forward(self, data, h_t, x_dyn_t):
        x_static = {nt: data[nt].x_static for nt in self.node_types}
        edge_static = {et: data[et].edge_attr_static for et in self.edge_types}

        for (src_type, rel, dst_type) in self.edge_types:
            key = f"{src_type}_{rel}_{dst_type}"
            mp = self.mp_modules[key]
            if isinstance(mp, GATv2CrossTypeMP):
                mp._set_context(h_src=h_t[src_type], h_dst=h_t[dst_type])
            else:
                mp._set_context(
                    h_src=h_t[src_type], h_dst=h_t[dst_type],
                    edge_attr_static=edge_static[(src_type, rel, dst_type)],
                    x_static_src=x_static[src_type], x_static_dst=x_static[dst_type])

        edge_index_dict = {et: data[et].edge_index for et in self.edge_types}
        x_dict = {nt: h_t[nt] for nt in self.node_types}
        messages = self.hetero_conv(x_dict, edge_index_dict)

        for nt in self.node_types:
            if nt not in messages:
                messages[nt] = torch.zeros((data[nt].num_nodes, self.msg_dim), device=h_t[nt].device)
            expected = h_t[nt].size(0)
            if messages[nt].size(0) != expected:
                raise RuntimeError(
                    f"Message shape mismatch for '{nt}': expected {expected} got {messages[nt].size(0)}")

        h_next = {}
        for nt in self.node_types:
            dyn_emb = self.dyn_norm[nt](self.dyn_proj[nt](x_dyn_t[nt]))
            msg_emb = self.msg_norm[nt](messages[nt])
            upd_in = torch.cat([dyn_emb, msg_emb], dim=-1)
            h_raw = self.update[nt](upd_in, h_t[nt])
            h_next[nt] = self.h_norm[nt](h_raw)
        return h_next


# ============================================================
# Main model: HeteroEncoderDecoderModel
# ============================================================

class NodeTransformerDecoder(nn.Module):
    """
    Transformer decoder over the time axis, applied independently per node.

    For each node type:
      1. Project [h_enc | anchor | rain_t | time_emb] → d_model  per timestep → [T, N, d_model]
      2. Reshape to [T, N, d_model] → treat as sequence of length T, batch size N
         via [T, N, d_model].permute(1,0,2) → [N, T, d_model] then into TransformerEncoder
         (PyTorch TransformerEncoder expects [T, B, d_model] so we pass N as batch)
      3. Project → delta, add anchor → absolute water level

    Each timestep attends to all others — captures rising/falling limb, autocorrelation,
    lag propagation effects that a pure MLP cannot.
    """
    def __init__(self, h_dim, d_model, nhead, num_layers, ffn_dim, T_max, dropout=0.1):
        super().__init__()
        self.T_max = T_max
        self.d_model = d_model
        # Input: h_enc[h_dim] + anchor[1] + rain[1] + time_emb[2] = h_dim + 4
        self.input_proj = nn.Linear(h_dim + 4, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, h_enc_nt, rain_t, anchor):
        """
        h_enc_nt: [N, h_dim]
        rain_t:   [T, N, 1]
        anchor:   [N, 1]

        Returns delta: [T, N, 1]  (caller adds anchor for absolute wl)
        """
        T, N, _ = rain_t.shape
        device = rain_t.device

        t_idx = torch.arange(T, device=device).float()
        t_emb = torch.stack([
            torch.sin(t_idx / self.T_max),
            torch.cos(t_idx / self.T_max),
        ], dim=-1)  # [T, 2]

        # Build per-timestep input: [T, N, h_dim+4]
        h_exp   = h_enc_nt.unsqueeze(0).expand(T, -1, -1)          # [T, N, h_dim]
        anc_exp = anchor.unsqueeze(0).expand(T, -1, -1)            # [T, N, 1]
        te_exp  = t_emb.unsqueeze(1).expand(-1, N, -1)             # [T, N, 2]
        inp = torch.cat([h_exp, anc_exp, rain_t, te_exp], dim=-1)  # [T, N, h_dim+4]

        # Project to d_model: [T, N, d_model]
        x = self.input_proj(inp)  # [T, N, d_model]

        # Transformer expects [B, T, d_model] with batch_first=True
        # Treat N nodes as batch (each node's T-length sequence is independent)
        x = x.permute(1, 0, 2)              # [N, T, d_model]
        x = self.transformer(x)             # [N, T, d_model]
        x = x.permute(1, 0, 2)              # [T, N, d_model]

        return self.out_proj(x)             # [T, N, 1]  (delta)


class HeteroEncoderDecoderModel(nn.Module):
    """
    Encoder-decoder flood model.

    Encoder: HeteroTransportCell × history_len steps (teacher-forced, B=1)
             h_dim=192, history_len=20 — richer basin state encoding

    Decoder: NodeTransformerDecoder — Transformer over time axis per node type.
             Each future timestep attends to all others → captures temporal dynamics,
             autocorrelation, lag effects. Predicts delta from last-known water level.

    Complexity: ~10× previous MLP decoder. Encoder also 2× wider/deeper.
    """

    def __init__(self, node_types, edge_types, node_static_dims, node_dyn_input_dims,
                 edge_static_dims, h_dim=192, msg_dim=128, hidden_dim=256,
                 decoder_hidden_dim=256, T_max=512,
                 dec_d_model=256, dec_nhead=8, dec_num_layers=4,
                 dec_ffn_dim=512, dec_dropout=0.1, dropout=0.0):
        super().__init__()
        self.node_types = node_types
        self.h_dim = h_dim
        self.T_max = T_max

        # Encoder GRU cell
        self.cell = HeteroTransportCell(
            node_types=node_types, edge_types=edge_types,
            node_static_dims=node_static_dims, node_dyn_input_dims=node_dyn_input_dims,
            edge_static_dims=edge_static_dims,
            h_dim=h_dim, msg_dim=msg_dim, hidden_dim=hidden_dim, dropout=dropout)

        # Transformer decoders — one per predicted node type
        _no_head = {"global", "ctx1d", "ctx2d"}
        self.decoders = nn.ModuleDict({
            nt: NodeTransformerDecoder(
                h_dim=h_dim, d_model=dec_d_model, nhead=dec_nhead,
                num_layers=dec_num_layers, ffn_dim=dec_ffn_dim,
                T_max=T_max, dropout=dec_dropout,
            )
            for nt in node_types if nt not in _no_head
        })
        self._pred_node_types = [nt for nt in node_types if nt not in _no_head]

    def init_hidden(self, data, device):
        h = {}
        for nt in self.node_types:
            N = data[nt].num_nodes
            h[nt] = torch.zeros(N, self.h_dim, device=device)
        return h

    def encode(self, data, y_hist_1d, y_hist_2d, rain_hist_2d, make_x_dyn, history_len, device):
        """Run encoder GRU × history_len steps. Returns h_enc: {nt: [N_nt, h_dim]}"""
        h = self.init_hidden(data, device)
        for k in range(history_len):
            x_dyn_t = make_x_dyn(y_hist_1d[k], y_hist_2d[k], rain_hist_2d[k])
            h = self.cell(data, h, x_dyn_t)
        return h

    def decode(self, h_enc, rain_future_2d, rain_1d_index, T_future, device,
               wl_anchor_1d=None, wl_anchor_2d=None):
        """
        h_enc:           {nt: [N_nt, h_dim]}
        rain_future_2d:  [T, N_2d, 1]
        wl_anchor_*:     [N_*, 1]  last known water level (t=H-1)

        Returns: {'oneD': [T, N_1d, 1], 'twoD': [T, N_2d, 1]}  (absolute wl)
        """
        preds = {}

        # 2D nodes
        N_2d = h_enc["twoD"].shape[0]
        anc_2d = wl_anchor_2d if wl_anchor_2d is not None else torch.zeros(N_2d, 1, device=device)
        delta_2d = self.decoders["twoD"](h_enc["twoD"], rain_future_2d, anc_2d)  # [T, N_2d, 1]
        preds["twoD"] = anc_2d.unsqueeze(0) + delta_2d

        # 1D nodes
        N_1d = h_enc["oneD"].shape[0]
        anc_1d = wl_anchor_1d if wl_anchor_1d is not None else torch.zeros(N_1d, 1, device=device)
        if rain_1d_index is not None:
            r_1d = rain_future_2d[:, rain_1d_index, :]                          # [T, N_1d, 1]
        else:
            r_1d = torch.zeros(T_future, N_1d, 1, device=device)
        delta_1d = self.decoders["oneD"](h_enc["oneD"], r_1d, anc_1d)           # [T, N_1d, 1]
        preds["oneD"] = anc_1d.unsqueeze(0) + delta_1d

        return preds

    def forward(self, data, y_hist_1d, y_hist_2d, rain_hist_2d, rain_future_2d,
                make_x_dyn, rain_1d_index, history_len, device):
        h_enc = self.encode(data, y_hist_1d, y_hist_2d, rain_hist_2d, make_x_dyn, history_len, device)
        wl_anchor_1d = y_hist_1d[-1]   # [N_1d, 1]
        wl_anchor_2d = y_hist_2d[-1]   # [N_2d, 1]
        T_future = rain_future_2d.shape[0]
        return self.decode(h_enc, rain_future_2d, rain_1d_index, T_future, device,
                           wl_anchor_1d=wl_anchor_1d, wl_anchor_2d=wl_anchor_2d)
