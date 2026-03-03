import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import MessagePassing, HeteroConv, GATv2Conv as _GATv2Conv


# ============================================================
# 0) Small utility: MLP blocks
# ============================================================
class MLP(nn.Module):
    """Simple MLP: Linear -> ReLU -> Linear (wireframe)."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# ============================================================
# 1) One relation-specific message passing module
#    This is where:
#      - static edge/node features influence coupling
#      - dynamic hidden states influence time-varying gating
# ============================================================
class StaticDynamicEdgeMP(MessagePassing):
    """
    One edge-type (relation) message passing module.

    For each directed edge j -> i, we compute an "effective coupling"
    that depends on:
      - static features (edge + endpoint node statics): time-invariant factors
      - dynamic hidden state (h_j, h_i): time-varying factors

    Then we form a message from source node j and aggregate into node i.

    Operates on [B*N, *] tensors — the batch dimension is folded into
    the node dimension so a single graph call covers all B samples.
    """
    def __init__(
        self,
        h_dim: int,
        node_static_dim_src: int,
        node_static_dim_dst: int,
        edge_static_dim: int,
        msg_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        aggr: str = "add",
    ):
        super().__init__(aggr=aggr)
        self.h_dim = h_dim
        self.msg_dim = msg_dim

        # ------------------------------------------------------------
        # (A) Static embedding: "Here we embed our static features"
        #
        # u_e = MLP_e([edge_static || src_static || dst_static])
        # This lets the model learn how slope/length/etc + endpoint attributes
        # affect the baseline coupling between nodes.
        # ------------------------------------------------------------
        self.edge_static_embed = MLP(
            in_dim=edge_static_dim + node_static_dim_src + node_static_dim_dst,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout
        )

        # ------------------------------------------------------------
        # (B) Static base weight: time-invariant coupling strength
        # b_e = softplus(w^T u_e)  (positive)
        # ------------------------------------------------------------
        self.base_weight = nn.Linear(hidden_dim, 1)

        # ------------------------------------------------------------
        # (C) Dynamic gate: "Here the model uses the current state"
        # g_e(t) = sigmoid(MLP_g([h_src || h_dst]))
        #
        # This allows the effective coupling to change with current water state,
        # representing e.g. saturation / thresholding / nonlinear flow.
        # ------------------------------------------------------------
        self.dynamic_gate = MLP(
            in_dim=2 * h_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            dropout=dropout
        )

        # ------------------------------------------------------------
        # (D) Payload: "This is the information that flows"
        # v_j(t) = MLP_v(h_src)
        # ------------------------------------------------------------
        self.payload = MLP(
            in_dim=h_dim,
            hidden_dim=hidden_dim,
            out_dim=msg_dim,
            dropout=dropout
        )

    def _set_context(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_attr_static: torch.Tensor,
        x_static_src: torch.Tensor,
        x_static_dst: torch.Tensor,
    ) -> None:
        """Store domain-specific context so forward() can use it (from HeteroConv)."""
        self._h_src = h_src
        self._h_dst = h_dst
        self._edge_attr_static = edge_attr_static
        self._x_static_src = x_static_src
        self._x_static_dst = x_static_dst

    def forward(
        self,
        x: torch.Tensor = None,               # Ignored (HeteroConv standard arg)
        edge_index: torch.Tensor = None,      # [2, E], src->dst
        **kwargs
    ) -> torch.Tensor:
        """
        Returns aggregated messages M_dst: [B*N_dst, msg_dim]

        Context (h_src, h_dst, edge_attr_static, x_static_src, x_static_dst)
        must be set via _set_context() before calling forward().
        """
        h_src = self._h_src
        h_dst = self._h_dst
        edge_attr_static = self._edge_attr_static
        x_static_src = self._x_static_src
        x_static_dst = self._x_static_dst

        return self.propagate(
            edge_index=edge_index,
            size=(h_src.size(0), h_dst.size(0)),
            h_src=h_src,
            h_dst=h_dst,
            edge_attr_static=edge_attr_static,
            x_static_src=x_static_src,
            x_static_dst=x_static_dst,
        )

    def message(
        self,
        h_src_j: torch.Tensor,                # [B*E, h_dim]   src hidden for each edge
        h_dst_i: torch.Tensor,                # [B*E, h_dim]   dst hidden for each edge
        edge_attr_static: torch.Tensor,       # [B*E, edge_static_dim]
        x_static_src_j: torch.Tensor,         # [B*E, node_static_dim_src]
        x_static_dst_i: torch.Tensor,         # [B*E, node_static_dim_dst]
    ) -> torch.Tensor:
        # ------------------------------------------------------------
        # 1) Static part: embed static edge + endpoint node features
        # ------------------------------------------------------------
        static_cat = torch.cat([edge_attr_static, x_static_src_j, x_static_dst_i], dim=-1)
        u_e = self.edge_static_embed(static_cat)                 # [B*E, hidden_dim]
        b_e = F.softplus(self.base_weight(u_e))                  # [B*E, 1] (positive)

        # ------------------------------------------------------------
        # 2) Dynamic part: gate based on current hidden states
        # ------------------------------------------------------------
        gate_in = torch.cat([h_src_j, h_dst_i], dim=-1)
        g_e = torch.sigmoid(self.dynamic_gate(gate_in))          # [B*E, 1]

        # ------------------------------------------------------------
        # 3) Payload: what is actually sent along the edge
        # ------------------------------------------------------------
        v = self.payload(h_src_j)                                # [B*E, msg_dim]

        # ------------------------------------------------------------
        # 4) Effective message: static base weight * dynamic gate * payload
        # ------------------------------------------------------------
        m = (b_e * g_e) * v                                      # [B*E, msg_dim]
        return m


# ============================================================
# 1b) GATv2-based cross-type message passing (1D channels -> 2D floodplain cells)
#
# Uses GATv2Conv attention to learn which channel hidden states are most
# relevant to each adjacent floodplain cell at each timestep.
# Wraps torch_geometric.nn.GATv2Conv in the same _set_context / forward
# interface as StaticDynamicEdgeMP so HeteroTransportCell can use it uniformly.
# ============================================================
class GATv2CrossTypeMP(MessagePassing):
    """
    GATv2-based message passing for directed 1D->2D cross-type edges.

    Computes attention-weighted messages from 1D channel hidden states
    to 2D floodplain cell hidden states.  The GATv2 attention score
    conditions on both src and dst hidden states, so each 2D cell can
    selectively attend to the most relevant channels.

    Output shape: [B*N_dst, msg_dim]
    """
    def __init__(
        self,
        h_dim: int,
        msg_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(aggr="add")
        self.h_dim = h_dim
        self.msg_dim = msg_dim
        self.heads = heads
        # GATv2Conv expects in_channels for (src, dst) separately
        self.gatv2 = _GATv2Conv(
            in_channels=(h_dim, h_dim),
            out_channels=msg_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
            add_self_loops=False,
        )

    def _set_context(self, h_src, h_dst, **kwargs):
        self._h_src = h_src
        self._h_dst = h_dst

    def forward(self, x=None, edge_index=None, **kwargs):
        # GATv2Conv takes (x_src, x_dst) tuple for bipartite graphs
        out = self.gatv2((self._h_src, self._h_dst), edge_index)
        return out  # [B*N_dst, msg_dim]


# ============================================================
# 2) One recurrent step over a hetero graph
#    This is where:
#      - dynamic data is injected into the graph
#      - past states are carried in hidden states
#      - messages are computed & aggregated
#      - hidden state updates (the "RNN" part)
# ============================================================
class HeteroTransportCell(nn.Module):
    """
    A single time step update:
        h_{t+1} = Update(h_t, dynamic_inputs_t, messages_t)

    Operates on batched inputs — hidden states and dynamic inputs have
    shape [B*N, *] where B is the batch size and N is the node count.
    The graph (edge_index, static features) is a PyG Batch of B copies
    of the same static graph, so edges are already offset correctly.
    """
    def __init__(
        self,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        node_static_dims: dict[str, int],
        node_dyn_input_dims: dict[str, int],   # how many dynamic inputs you feed each node type per step
        edge_static_dims: dict[tuple[str, str, str], int],
        h_dim: int = 64,
        msg_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.h_dim = h_dim
        self.msg_dim = msg_dim

        # ------------------------------------------------------------
        # A) Hetero message passing blocks (one per relation)
        # ------------------------------------------------------------
        conv_dict = {}
        for (src, rel, dst) in edge_types:
            if rel == "oneDtwoD":
                # GATv2Conv for 1D->2D: attention over channel hidden states
                mp = GATv2CrossTypeMP(
                    h_dim=h_dim,
                    msg_dim=msg_dim,
                    heads=4,
                    dropout=dropout,
                )
            else:
                mp = StaticDynamicEdgeMP(
                    h_dim=h_dim,
                    node_static_dim_src=node_static_dims[src],
                    node_static_dim_dst=node_static_dims[dst],
                    edge_static_dim=edge_static_dims[(src, rel, dst)],
                    msg_dim=msg_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    aggr="add",
                )
            conv_dict[(src, rel, dst)] = mp

        # Store both dict-based and module dict for HeteroConv usage
        self.mp_modules = nn.ModuleDict({
            f"{src}_{rel}_{dst}": mp for (src, rel, dst), mp in conv_dict.items()
        })
        self.edge_types = list(conv_dict.keys())  # Store for iteration

        # Use HeteroConv to orchestrate message passing (for clean PyG integration)
        self.hetero_conv = HeteroConv(conv_dict, aggr="sum")

        # ------------------------------------------------------------
        # B) Dynamic input projection per node type
        #    "This is where the dynamic data is passed to the graph"
        # ------------------------------------------------------------
        self.dyn_proj = nn.ModuleDict({
            nt: nn.Linear(node_dyn_input_dims[nt], msg_dim) for nt in node_types
        })

        # ------------------------------------------------------------
        # C) Recurrent update per node type
        #    GRUCell operates on [B*N, *] directly.
        # ------------------------------------------------------------
        self.update = nn.ModuleDict({
            nt: nn.GRUCell(input_size=2 * msg_dim, hidden_size=h_dim) for nt in node_types
        })

    def forward(
        self,
        data: HeteroData,
        h_t: dict[str, torch.Tensor],
        x_dyn_t: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Perform ONE timestep update over a batch of B graphs.

        data: PyG Batch of B copies of the static graph.
              data[nt].x_static              [B*N_nt, node_static_dim]
              data[etype].edge_index         [2, B*E]  (offsets applied by Batch)
              data[etype].edge_attr_static   [B*E, edge_static_dim]

        h_t[nt]:     [B*N_nt, h_dim]   — hidden state (memory) from previous step
        x_dyn_t[nt]: [B*N_nt, dyn_dim] — dynamic observed input at this timestep

        Returns:
          h_{t+1} dict, same shapes as h_t
        """
        # Grab static node features dict (constant over time, replicated B times by Batch)
        x_static = {nt: data[nt].x_static for nt in self.node_types}

        # Grab edge static attributes per edge type
        edge_static = {et: data[et].edge_attr_static for et in self.edge_types}

        # ------------------------------------------------------------
        # 1) Inject context into each MP module, then call HeteroConv.
        #    All tensors are already in [B*N, *] / [B*E, *] space.
        # ------------------------------------------------------------
        for (src_type, rel, dst_type) in self.edge_types:
            key = f"{src_type}_{rel}_{dst_type}"
            mp = self.mp_modules[key]
            if rel == "oneDtwoD":
                mp._set_context(
                    h_src=h_t[src_type],
                    h_dst=h_t[dst_type],
                )
            else:
                mp._set_context(
                    h_src=h_t[src_type],
                    h_dst=h_t[dst_type],
                    edge_attr_static=edge_static[(src_type, rel, dst_type)],
                    x_static_src=x_static[src_type],
                    x_static_dst=x_static[dst_type],
                )

        # Build edge_index_dict and x_dict for HeteroConv (standard PyG interface).
        # Pass h_t as x so HeteroConv correctly infers per-type node counts [B*N_nt, *].
        edge_index_dict = {et: data[et].edge_index for et in self.edge_types}
        x_dict = {nt: h_t[nt] for nt in self.node_types}

        # Call HeteroConv — returns messages[node_type] = [B*N_dst, msg_dim]
        messages = self.hetero_conv(x_dict, edge_index_dict)

        # Ensure all destination node types have messages
        for nt in self.node_types:
            if nt not in messages:
                messages[nt] = torch.zeros((data[nt].num_nodes, self.msg_dim), device=h_t[nt].device)

            expected_n_dst = h_t[nt].size(0)
            if messages[nt].size(0) != expected_n_dst:
                raise RuntimeError(
                    f"Message shape mismatch for node type '{nt}': "
                    f"expected {expected_n_dst} nodes but got {messages[nt].size(0)} "
                    f"(msg shape: {messages[nt].shape})"
                )

        # ------------------------------------------------------------
        # 2) Inject DYNAMIC inputs and run GRU update.
        #    All in [B*N, *] space — GRUCell handles this natively.
        # ------------------------------------------------------------
        h_next = {}
        for nt in self.node_types:
            dyn_emb = self.dyn_proj[nt](x_dyn_t[nt])  # [B*N, msg_dim]
            msg_emb = messages[nt]                      # [B*N, msg_dim]
            upd_in = torch.cat([dyn_emb, msg_emb], dim=-1)  # [B*N, 2*msg_dim]
            h_next[nt] = self.update[nt](upd_in, h_t[nt])   # [B*N, h_dim]

        return h_next


# ============================================================
# 3) Full autoregressive model (warm start + rollout)
#    This is where:
#      - we predict future water level
#      - we feed predictions forward (autoregression)
# ============================================================
class FloodAutoregressiveHeteroModel(nn.Module):
    """
    High-level model:
      - HeteroTransportCell: evolves hidden state over time
      - Heads: maps hidden state -> predicted water level for both 1D and 2D nodes

    All operations are vectorized over the batch dimension B.
    Hidden states have shape [B*N, h_dim]; the static graph is replicated
    B times via PyG's Batch so message passing covers all samples at once.
    """
    def __init__(
        self,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        node_static_dims: dict[str, int],
        node_dyn_input_dims: dict[str, int],
        edge_static_dims: dict[tuple[str, str, str], int],
        h_dim: int = 64,
        msg_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.h_dim = h_dim

        # The recurrent graph cell
        self.cell = HeteroTransportCell(
            node_types=node_types,
            edge_types=edge_types,
            node_static_dims=node_static_dims,
            node_dyn_input_dims=node_dyn_input_dims,
            edge_static_dims=edge_static_dims,
            h_dim=h_dim,
            msg_dim=msg_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ------------------------------------------------------------
        # Prediction heads (one per node type):
        # Each head maps hidden state -> predicted water level.
        # Operates on [B*N, h_dim] -> [B*N, 1].
        # ------------------------------------------------------------
        self.heads = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(h_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for nt in node_types
        })

    def _make_batched_graph(self, data: HeteroData, B: int) -> HeteroData:
        """
        Replicate the static graph B times using PyG's Batch mechanism.
        Returns a batched HeteroData where:
          - node features: [B*N_nt, *]
          - edge indices:  [2, B*E] with correct per-copy offsets
          - edge features: [B*E, *]

        num_nodes must be set explicitly so PyG can compute edge index offsets.
        """
        # Ensure num_nodes is set on each node type store so Batch can offset edges
        for nt in self.node_types:
            data[nt].num_nodes = data[nt].x_static.size(0)
        return Batch.from_data_list([data] * B)

    def init_hidden(self, data: HeteroData, B: int, device: torch.device) -> dict[str, torch.Tensor]:
        """
        Initialize hidden state for a batch of B samples.
        Returns h[nt]: [B*N_nt, h_dim]
        """
        h = {}
        for nt in self.node_types:
            N = data[nt].num_nodes
            h[nt] = torch.zeros((B * N, self.cell.h_dim), device=device)
        return h

    def predict_water_levels(
        self,
        h: dict[str, torch.Tensor],
        B: int,
        node_counts: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        """
        Decode hidden states into water level predictions.

        h[nt]: [B*N_nt, h_dim]
        Returns: {'oneD': [B, N_1d, 1], 'twoD': [B, N_2d, 1]}
        """
        out = {}
        for nt in self.heads:
            N = node_counts[nt]
            pred_flat = self.heads[nt](h[nt])          # [B*N, 1]
            out[nt] = pred_flat.view(B, N, 1)           # [B, N, 1]
        return out

    def forward_unroll(
        self,
        data: HeteroData,
        y_hist_1d: torch.Tensor,       # [B, H, N_1d, 1]
        y_hist_2d: torch.Tensor,       # [B, H, N_2d, 1]
        rain_hist: torch.Tensor,       # [B, H, N_2d, R]
        rain_future: torch.Tensor,     # [B, T, N_2d, R]
        make_x_dyn,                    # function(y_pred: dict, rain_2d, data) -> x_dyn dict
        rollout_steps: int,
        device: torch.device,
        batched_data: HeteroData = None,  # Pre-built batched graph (optional, avoids rebuild each call)
        use_grad_checkpoint: bool = False,  # Trades recompute for memory at long horizons
    ) -> dict[str, torch.Tensor]:
        """
        Vectorized forward pass over a batch of B samples.

        Run:
          1) Warm start (teacher forcing) for H steps using true y
          2) Autoregressive rollout for rollout_steps using predicted y

        make_x_dyn signature:
          (y_pred: {'oneD': [B*N_1d, 1], 'twoD': [B*N_2d, 1]},
           rain_2d: [B*N_2d, R],
           data: batched HeteroData)
          -> {'oneD': [B*N_1d, dyn_1d], 'twoD': [B*N_2d, dyn_2d]}

        Returns:
            {'oneD': [B, rollout_steps, N_1d, 1],
             'twoD': [B, rollout_steps, N_2d, 1]}
        """
        B = y_hist_1d.size(0)
        N_1d = y_hist_1d.size(2)
        N_2d = y_hist_2d.size(2)
        node_counts = {'oneD': N_1d, 'twoD': N_2d}

        # Use pre-built batched graph if provided, otherwise build it now
        if batched_data is None:
            batched_data = self._make_batched_graph(data, B)

        h = self.init_hidden(data, B, device=device)

        # ------------------------------------------------------------
        # (1) Warm start: feed true past states so hidden state learns history
        # Reshape [B, H, N, F] inputs to [B*N, F] for each timestep k
        # ------------------------------------------------------------
        H = y_hist_1d.size(1)
        for k in range(H):
            # [B, N, 1] -> [B*N, 1]
            y1d_k = y_hist_1d[:, k, :, :].reshape(B * N_1d, 1)
            y2d_k = y_hist_2d[:, k, :, :].reshape(B * N_2d, 1)
            r_k   = rain_hist[:, k, :, :].reshape(B * N_2d, -1)

            y_true_k = {'oneD': y1d_k, 'twoD': y2d_k}
            x_dyn_t = make_x_dyn(y_true_k, r_k, batched_data)
            h = self.cell(batched_data, h, x_dyn_t)

        # Start rollout from last observed water levels (flattened)
        y_t = {
            'oneD': y_hist_1d[:, -1, :, :].reshape(B * N_1d, 1),
            'twoD': y_hist_2d[:, -1, :, :].reshape(B * N_2d, 1),
        }

        preds_1d = []
        preds_2d = []

        # ------------------------------------------------------------
        # (2) Autoregressive rollout:
        #     - predict next water level for all B samples at once
        #     - feed predictions forward as inputs at next step
        # Gradient checkpointing (use_grad_checkpoint=True) trades recompute
        # for memory at long horizons — each cell step recomputes its activations
        # during backward instead of storing them.
        # ------------------------------------------------------------
        for t in range(rollout_steps):
            y_next = self.predict_water_levels(h, B, node_counts)
            # y_next['oneD']: [B, N_1d, 1], y_next['twoD']: [B, N_2d, 1]

            preds_1d.append(y_next['oneD'])   # [B, N_1d, 1]
            preds_2d.append(y_next['twoD'])   # [B, N_2d, 1]

            # Flatten for cell input
            r_next = rain_future[:, t, :, :].reshape(B * N_2d, -1)
            y_flat = {
                'oneD': y_next['oneD'].reshape(B * N_1d, 1),
                'twoD': y_next['twoD'].reshape(B * N_2d, 1),
            }
            x_dyn_next = make_x_dyn(y_flat, r_next, batched_data)
            if use_grad_checkpoint and self.training:
                # checkpoint requires all inputs to be tensors; pass h values as a flat list
                h_1d, h_2d = h['oneD'], h['twoD']
                def _cell_step(h_1d, h_2d, dyn_1d, dyn_2d):
                    return self.cell(batched_data,
                                     {'oneD': h_1d, 'twoD': h_2d},
                                     {'oneD': dyn_1d, 'twoD': dyn_2d})
                h_out = grad_checkpoint(_cell_step, h_1d, h_2d,
                                        x_dyn_next['oneD'], x_dyn_next['twoD'],
                                        use_reentrant=False)
                h = h_out
            else:
                h = self.cell(batched_data, h, x_dyn_next)
            y_t = y_flat

        return {
            'oneD': torch.stack(preds_1d, dim=1),  # [B, rollout_steps, N_1d, 1]
            'twoD': torch.stack(preds_2d, dim=1),  # [B, rollout_steps, N_2d, 1]
        }
