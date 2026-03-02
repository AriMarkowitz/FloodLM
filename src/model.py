import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, HeteroConv


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
        Returns aggregated messages M_dst: [N_dst, msg_dim]

        Context (h_src, h_dst, edge_attr_static, x_static_src, x_static_dst) 
        must be set via _set_context() before calling forward().
        
        This design allows HeteroConv to call modules with standard signature.
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
        h_src_j: torch.Tensor,                # [E, h_dim]   src hidden for each edge
        h_dst_i: torch.Tensor,                # [E, h_dim]   dst hidden for each edge
        edge_attr_static: torch.Tensor,       # [E, edge_static_dim]
        x_static_src_j: torch.Tensor,         # [E, node_static_dim_src]
        x_static_dst_i: torch.Tensor,         # [E, node_static_dim_dst]
    ) -> torch.Tensor:
        # ------------------------------------------------------------
        # 1) Static part: embed static edge + endpoint node features
        # ------------------------------------------------------------
        static_cat = torch.cat([edge_attr_static, x_static_src_j, x_static_dst_i], dim=-1)
        u_e = self.edge_static_embed(static_cat)                 # [E, hidden_dim]
        b_e = F.softplus(self.base_weight(u_e))                  # [E, 1] (positive)

        # ------------------------------------------------------------
        # 2) Dynamic part: gate based on current hidden states
        # ------------------------------------------------------------
        gate_in = torch.cat([h_src_j, h_dst_i], dim=-1)
        g_e = torch.sigmoid(self.dynamic_gate(gate_in))          # [E, 1]

        # ------------------------------------------------------------
        # 3) Payload: what is actually sent along the edge
        # ------------------------------------------------------------
        v = self.payload(h_src_j)                                # [E, msg_dim]

        # ------------------------------------------------------------
        # 4) Effective message: static base weight * dynamic gate * payload
        # ------------------------------------------------------------
        m = (b_e * g_e) * v                                      # [E, msg_dim]
        return m


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

    where messages_t are produced by hetero message passing using:
      - static graph (edge_index)
      - static features (node & edge)
      - current hidden states h_t (dynamic memory)
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
        #
        # Example dynamic inputs for a node type:
        #   [current water level estimate, rainfall_t, rainfall_t-1, ...]
        # We project those into msg_dim so they combine nicely with messages.
        # ------------------------------------------------------------
        self.dyn_proj = nn.ModuleDict({
            nt: nn.Linear(node_dyn_input_dims[nt], msg_dim) for nt in node_types
        })

        # ------------------------------------------------------------
        # C) Recurrent update per node type
        #    This is where "past states are embedded":
        #    h_t is the memory of previous steps, updated each time step.
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
        Perform ONE timestep update.

        data contains STATIC graph structure & static features:
          data[nt].x_static              [N_nt, node_static_dim]
          data[etype].edge_index         [2, E]
          data[etype].edge_attr_static   [E, edge_static_dim]

        h_t is the DYNAMIC hidden state (memory) from previous step:
          h_t[nt]                        [N_nt, h_dim]

        x_dyn_t is the DYNAMIC observed input at this timestep:
          x_dyn_t[nt]                    [N_nt, dyn_dim_nt]

        Returns:
          h_{t+1} dict
        """
        # Grab static node features dict (constant over time)
        x_static = {nt: data[nt].x_static for nt in self.node_types}

        # Grab edge static attributes per edge type
        edge_static = {et: data[et].edge_attr_static for et in self.edge_types}

        # ------------------------------------------------------------
        # 1) Inject context into each MP module, then call HeteroConv
        #    HeteroConv will call each module's forward() with standard args
        #    But our modules can access the injected context during propagate()
        # ------------------------------------------------------------
        for (src_type, rel, dst_type) in self.edge_types:
            key = f"{src_type}_{rel}_{dst_type}"
            mp = self.mp_modules[key]
            mp._set_context(
                h_src=h_t[src_type],
                h_dst=h_t[dst_type],
                edge_attr_static=edge_static[(src_type, rel, dst_type)],
                x_static_src=x_static[src_type],
                x_static_dst=x_static[dst_type],
            )

        # Build edge_index_dict and x_dict for HeteroConv (standard PyG interface)
        edge_index_dict = {et: data[et].edge_index for et in self.edge_types}
        x_dict = {nt: torch.zeros((data[nt].num_nodes, 1), device=h_t[nt].device) for nt in self.node_types}  # Dummy x

        # Call HeteroConv which orchestrates all message passing
        # Returns: dict keyed by DESTINATION NODE TYPE (not edge type!)
        # messages[node_type] = [N_dst_type, msg_dim]
        # HeteroConv already aggregates messages from ALL incoming edge types per destination
        messages = self.hetero_conv(x_dict, edge_index_dict)
        
        # Ensure all destination node types have messages (add zero tensors if missing)
        for nt in self.node_types:
            if nt not in messages:
                messages[nt] = torch.zeros((data[nt].num_nodes, self.msg_dim), device=h_t[nt].device)
            
            # Validate message shape matches destination node count
            expected_n_dst = h_t[nt].size(0)
            if messages[nt].size(0) != expected_n_dst:
                raise RuntimeError(
                    f"Message shape mismatch for node type '{nt}': "
                    f"expected {expected_n_dst} nodes but got {messages[nt].size(0)} "
                    f"(msg shape: {messages[nt].shape})"
                )

        # ------------------------------------------------------------
        # 2) Inject DYNAMIC inputs (rain, water estimate, etc.)
        #    "This is where dynamic data enters the model each timestep"
        # ------------------------------------------------------------
        h_next = {}
        for nt in self.node_types:
            dyn_emb = self.dyn_proj[nt](x_dyn_t[nt])  # [N, msg_dim]
            
            # Get aggregated message for this node type
            msg_emb = messages[nt]  # [N, msg_dim]
            
            # Validate shapes match
            if msg_emb.size(0) != dyn_emb.size(0):
                raise RuntimeError(
                    f"Node count mismatch for '{nt}': "
                    f"expected {msg_emb.size(0)} nodes but dyn_emb has {dyn_emb.size(0)} "
                    f"(msg_emb: {msg_emb.shape}, dyn_emb: {dyn_emb.shape})"
                )
            
            upd_in = torch.cat([dyn_emb, msg_emb], dim=-1)         # [N, 2*msg_dim]

            # --------------------------------------------------------
            # 3) Recurrent update: embeds history into h
            #    "Here the model embeds past states"
            # --------------------------------------------------------
            h_next[nt] = self.update[nt](upd_in, h_t[nt])           # [N, h_dim]

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
      - Head: maps hidden state -> predicted water level (or delta)
    """
    def __init__(
        self,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        node_static_dims: dict[str, int],
        node_dyn_input_dims: dict[str, int],
        edge_static_dims: dict[tuple[str, str, str], int],
        pred_node_type: str,    # node type whose water level we predict
        h_dim: int = 64,
        msg_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        predict_delta: bool = True,
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.pred_node_type = pred_node_type
        self.predict_delta = predict_delta

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
        # Prediction head:
        # "Here we predict our target future water level"
        # ------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def init_hidden(self, data: HeteroData, device: torch.device) -> dict[str, torch.Tensor]:
        """Initialize hidden state (this is the memory of past states)."""
        h = {}
        for nt in self.node_types:
            h[nt] = torch.zeros((data[nt].num_nodes, self.cell.h_dim), device=device)
        return h

    def predict_water_level(self, h: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Decode hidden state at current step into a prediction for the target nodes.
        """
        return self.head(h[self.pred_node_type])  # [N_pred, 1]

    def forward_unroll(
        self,
        data: HeteroData,
        # Provided by you (depends on how you store your data):
        y_hist_true: torch.Tensor,     # [H, N_pred, 1] true water levels for warm start
        rain_hist: torch.Tensor,       # [H, N_pred, R] forcing for warm start
        rain_future: torch.Tensor,     # [T, N_pred, R] forcing for rollout
        make_x_dyn,                    # function: builds x_dyn_t dict from (y_input, rain_input)
        rollout_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Run:
          1) Warm start (teacher forcing) for H steps using true y
          2) Autoregressive rollout for rollout_steps using predicted y

        Returns predicted water levels:
          y_preds: [rollout_steps, N_pred, 1]
        """
        h = self.init_hidden(data, device=device)

        # ------------------------------------------------------------
        # (1) Warm start: feed true past states so hidden state learns history
        # ------------------------------------------------------------
        H = y_hist_true.size(0)
        for k in range(H):
            y_in = y_hist_true[k].to(device)  # true water
            r_in = rain_hist[k].to(device)    # observed rain
            x_dyn_t = make_x_dyn(y_in, r_in, data)  # <-- YOU define for all node types
            h = self.cell(data, h, x_dyn_t)         # <-- hidden state updated

        # We start rollout from the last observed water level
        y_t = y_hist_true[-1].to(device)

        preds = []

        # ------------------------------------------------------------
        # (2) Autoregressive rollout:
        #     - predict next water level
        #     - feed prediction forward as input at next step
        #     - use known future rainfall forecast
        # ------------------------------------------------------------
        for t in range(rollout_steps):
            # "Here we predict our target future water level"
            delta_or_abs = self.predict_water_level(h)  # [N_pred, 1]

            if self.predict_delta:
                y_next = y_t + delta_or_abs
            else:
                y_next = delta_or_abs

            preds.append(y_next)

            # "Here we pass dynamic data (predicted y + forecast rain) into the graph"
            r_next = rain_future[t].to(device)
            x_dyn_next = make_x_dyn(y_next, r_next, data)

            # update hidden state forward one step
            h = self.cell(data, h, x_dyn_next)

            # advance
            y_t = y_next

        return torch.stack(preds, dim=0)  # [rollout_steps, N_pred, 1]