import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttentionBlock(nn.Module):
	"""
	Multi-Head Attention Block
	"""
	def __init__(self, input_dim, hidden_dim, num_heads):
		super().__init__()
		self.num_heads = num_heads
		self.hidden_dim = hidden_dim
		self.head_dim = hidden_dim // num_heads  # 128/4=32

		# Q,K,V transformation matrices
		self.q_linear = nn.Linear(input_dim, hidden_dim, bias=False)
		self.k_linear = nn.Linear(input_dim, hidden_dim, bias=False)
		self.v_linear = nn.Linear(input_dim, hidden_dim, bias=False)
		self.o_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

		self.attn_dropout = nn.Dropout(0.1)

	# Scaled dot product attention
	def scaled_dot_product_attention(self, Q, K, V, mask=None):
		scale = math.sqrt(self.head_dim) + 1e-8
		
		attn_scores = torch.matmul(Q, K.transpose(-2 ,-1)    # Q*K^T
		) / scale                                            # scale

		if mask is not None:
			attn_scores = attn_scores.masked_fill(
				mask == 0, float("-inf")
			)

		attn_weights = F.softmax(attn_scores, dim=-1)    # [bs, num_heads, seq_len, seq_len]
		attn_weights = self.attn_dropout(attn_weights)
		out = torch.matmul(attn_weights, V)

		return out, attn_weights

	def forward(self, x):
		batch_size = x.size(0)

		# Wq, Wk, Wv transformations
		Q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim)
		K = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim)
		V = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim)
		
		Q = Q.transpose(1, 2)
		K = K.transpose(1, 2)
		V = V.transpose(1, 2)

		out, _ = self.scaled_dot_product_attention(Q, K, V)
		out = out.transpose(1, 2).contiguous()
		out = out.view(batch_size, -1, self.hidden_dim)

		out = self.o_linear(out)    # [bs, seq_len, hidden_dim]

		return out

class AtomicPairAttentionLayer(nn.Module):
	"""Atomic pair attention layer"""
	def __init__(self, hidden_dim, num_heads, dropout_rate=0.2):
		super().__init__()
		self.mha_attention = MultiHeadAttentionBlock(hidden_dim, hidden_dim, num_heads)
		self.norm1 = nn.LayerNorm(hidden_dim)
		self.norm2 = nn.LayerNorm(hidden_dim)
		self.dropout = nn.Dropout(dropout_rate)

		self.mlp = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim * 2),
			nn.SiLU(),
			nn.Dropout(dropout_rate),
			nn.Linear(hidden_dim * 2, hidden_dim)
		)
		
	def forward(self, x):            # x: [bs, num_atoms, hidden_dim]
		# residual connection
		x_norm = self.norm1(x)    
		attn_out = self.mha_attention(x_norm)
		x = x + self.dropout(attn_out)    
		
		# residual connection
		x_norm = self.norm2(x)      
		mlp_out = self.mlp(x_norm)    
		x = x + self.dropout(mlp_out)    # x: [bs, num_atoms, hidden_dim]

		return x

class FinalAttentionLayer(nn.Module):
	"""Output layer"""
	def __init__(self, hidden_dim, num_heads, dropout_rate=0.2):
		super().__init__()
		self.num_heads = num_heads
		self.head_dim = hidden_dim // num_heads

		self.mha_attention = MultiHeadAttentionBlock(hidden_dim, hidden_dim, num_heads)
		self.norm1 = nn.LayerNorm(hidden_dim)
		self.norm2 = nn.LayerNorm(hidden_dim)
		self.dropout = nn.Dropout(dropout_rate)

		self.fusion_norm = nn.LayerNorm(hidden_dim)

		self.mlp = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim * 2),
			nn.SiLU(),
			nn.Dropout(dropout_rate),
			nn.Linear(hidden_dim * 2, hidden_dim)
		)

		self.output_mlp = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim // 4),            # 128 -> 32
			nn.SiLU(),
			nn.Dropout(dropout_rate),
			nn.Linear(hidden_dim // 4, hidden_dim // 8),    # 32 -> 16
			nn.SiLU(), 
			nn.Dropout(dropout_rate),
			nn.Linear(hidden_dim // 8, 1),                # 16 -> 1
			nn.Sigmoid()
		)

	def forward(self, x, angles):                # x: [bs, num_atoms, hidden_dim]
		# residual connection
		x_norm = self.norm1(x)
		attn_out = self.mha_attention(x_norm)
		x = x + self.dropout(attn_out)

		# residual connection
		x_norm = self.norm2(x)
		mlp_out = self.mlp(x_norm)
		x = x + self.dropout(mlp_out)        # x: [bs, num_atoms, hidden_dim]

		angles = angles.squeeze(-1)
		angles = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
		angles_feats = angles.unsqueeze(-1)
		x_feats = x.unsqueeze(-2)

		# Matrix multiplication outer product
		angle_x_product= torch.matmul(angles_feats, x_feats)    

		# Dimension compression
		x_angle_fused = angle_x_product.sum(dim=2)    # [bs, num_atoms, 128]
		x_angle_fused = self.fusion_norm(x_angle_fused)

		# residual connection
		x = x + x_angle_fused            # [bs, num_atoms, hidden_dim]

		out = self.output_mlp(x)            # [bs*num_atoms, 1]

		return out

class AtomicPairTransformer(nn.Module):
	"""Learns per-atom-pair importance weights via self-attention with angle-aware fusion."""
	def __init__(self, 
				hidden_dim=128,        # Hidden layer dimension
				num_heads=4,        # Number of heads
				num_layers=4,
				dropout_rate=0.2):
		super().__init__()
		self.input_proj = nn.Linear(hidden_dim, hidden_dim)    # 128 -> 128
		
		# Intermediate layers
		self.intermediate_layers = nn.ModuleList([
			AtomicPairAttentionLayer(
				hidden_dim=hidden_dim,
				num_heads=num_heads,
				dropout_rate=dropout_rate
			) for _ in range(num_layers-1)
		])
		# Final output layer
		self.final_layer = FinalAttentionLayer(
			hidden_dim=hidden_dim,
			num_heads=num_heads,
			dropout_rate=dropout_rate
		)

	def forward(self, x, sigma, angles):
		# Input projection and sigma modulation
		x = self.input_proj(x)                # [bs, num_atoms, 12] -> [bs, num_atoms, 128]

		sigma = torch.clamp(sigma, min=1.2, max=6.0)
		sigma = sigma.unsqueeze(-1).expand(-1, -1, x.size(-1))    # [bs, num_atoms] -> [bs, num_atoms, 128]
		x = x * sigma                                              # [bs, num_atoms, 128] * [bs, num_atoms, 128]

		# Iteration layer
		for block in self.intermediate_layers:
			x = block(x)            # 128 -> 128
		
		# final layer
		out = self.final_layer(x, angles)    # 128 -> 1
		return out


class ACANet(nn.Module):
	"""
	Atomic Context Aggregation Network.

	Fuses atom-pair features with neighbor context via cross-attention
	to predict per-pair σ (radius), c (temperature), and b (baseline) parameters.
	"""
	def __init__(self, atom_feat_dim, bond_feat_dim, hidden_dim):
		super().__init__()
		# Projection layers
		self.interaction_proj = nn.Linear(atom_feat_dim, hidden_dim)
		self.neighbor_proj = nn.Linear(atom_feat_dim + bond_feat_dim, hidden_dim)

		# Norm layers
		self.norm_neigh1 = nn.LayerNorm(hidden_dim)
		self.norm_neigh2 = nn.LayerNorm(hidden_dim)

		self.fused_norm = nn.LayerNorm(hidden_dim)
		
		self.norm_1 = nn.LayerNorm(hidden_dim)
		self.norm_2 = nn.LayerNorm(hidden_dim)
		self.norm_3 = nn.LayerNorm(hidden_dim)

		self.norm_h1 = nn.LayerNorm(hidden_dim)
		self.norm_h2 = nn.LayerNorm(hidden_dim)
		self.norm_h3 = nn.LayerNorm(hidden_dim)
		self.norm_h4 = nn.LayerNorm(hidden_dim)

		# Transformations
		self.Linear_1 = nn.Linear(hidden_dim, hidden_dim)
		self.Linear_2 = nn.Linear(hidden_dim, hidden_dim)
		self.Linear_3 = nn.Linear(hidden_dim, hidden_dim)
		self.Linear_4 = nn.Linear(hidden_dim, hidden_dim)

		self.Linear_21 = nn.Linear(hidden_dim, hidden_dim)
		self.Linear_22 = nn.Linear(hidden_dim, hidden_dim)
		self.Linear_23 = nn.Linear(hidden_dim, hidden_dim)
		self.Linear_24 = nn.Linear(hidden_dim, hidden_dim)

		self.dropout = nn.Dropout(0.2)

		# Parameter prediction heads
		self.sigma_predictor = self._build_pred_head(hidden_dim)
		self.c_predictor = self._build_pred_head(hidden_dim)
		self.b_predictor = self._build_pred_head(hidden_dim)

		# Initialize learnable scaling factors
		self.c_scale = nn.Parameter(torch.tensor(0.5))
		self.b_scale = nn.Parameter(torch.tensor(0.5))

	def _build_pred_head(self, hidden_dim):
		return nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim//2),        # 128 -> 64
			nn.SiLU(),
			nn.Dropout(0.1),
			nn.BatchNorm1d(hidden_dim//2),
			nn.Linear(hidden_dim//2, hidden_dim//4),    # 64 -> 32
			nn.SiLU(), 
			nn.Dropout(0.1),
			nn.BatchNorm1d(hidden_dim//4),
			nn.Linear(hidden_dim//4, 1),                # 32 -> 1
			nn.Sigmoid()
		)

	def forward(self, x1, x2, x1_neigh, x2_neigh, mask):    # x1,x2: [bs, m, 12]
		# Atom features projection 
		x1_o = self.interaction_proj(x1)                    # [bs, m, 12] -> [bs, m, hd]
		x2_o = self.interaction_proj(x2)

		# Neighbor projection and fused
		x1_neigh = self.neighbor_proj(x1_neigh)            # [bs, m, 6, 14] -> [bs, m, 6, hd]
		x2_neigh = self.neighbor_proj(x2_neigh)

		# neigh padding mask
		x1_neigh_mask = (x1_neigh.sum(dim=-1) != 0)        # [bs, m, 6]
		x2_neigh_mask = (x2_neigh.sum(dim=-1) != 0)

		# Hadamard product
		x1_neigh_inter = x1_o.unsqueeze(2) * x1_neigh        # [bs, m, 1, hd] * [bs, m, 6, hd] = [bs, m, 6, hd]
		x2_neigh_inter = x2_o.unsqueeze(2) * x2_neigh

		# Aggregate neighbor information based on mask
		x1_neigh_fused = (x1_neigh_inter * x1_neigh_mask.unsqueeze(-1)).sum(dim=2)    # [bs, m, hd]
		x2_neigh_fused = (x2_neigh_inter * x2_neigh_mask.unsqueeze(-1)).sum(dim=2)

		# Mean pooling
		x1_neigh_fused_mean = x1_neigh_fused / (x1_neigh_mask.sum(dim=2, keepdim=True) + 1e-6)    # [bs, m, hd]
		x2_neigh_fused_mean = x2_neigh_fused / (x2_neigh_mask.sum(dim=2, keepdim=True) + 1e-6)

		# Norm
		x1_o = self.norm_1(x1_o)                # [bs, m, 1, hd] -> [bs, m, hd]
		x2_o = self.norm_1(x2_o)

		x1_neigh_fused_mean = self.norm_neigh1(x1_neigh_fused_mean)
		x2_neigh_fused_mean = self.norm_neigh1(x2_neigh_fused_mean)

		# Residual connection
		x1_fused = x1_o + x1_neigh_fused_mean    # [bs, m, hd] + [bs, m, hd] = [bs, m, hd]
		x2_fused = x2_o + x2_neigh_fused_mean

		# Norm
		x1_fused = self.fused_norm(x1_fused)
		x2_fused = self.fused_norm(x2_fused)
		
		# linear transformations
		H1 = self.Linear_1(x1_fused)            # H1,H2,H3,H4 shape :[bs, m, hidden_dim]
		H2 = self.Linear_2(x2_fused) 
		H3 = self.Linear_3(x2_fused)
		H4 = self.Linear_4(x2_fused)
		
		# Cross attention
		H1 = self.Linear_21(self.norm_h1(H1))
		H2 = self.Linear_22(self.norm_h2(H2*H1))
		H3 = self.Linear_23(self.norm_h3(H3*H1))
		H4 = self.Linear_24(self.norm_h4(H4*H1))
		
		sigma_input = H1*H2        # sigma [bs, m, hidden_dim]
		c_input = H1*H3            # c
		b_input = H1*H4            # b
		
		batch_size, seq_len, hidden_dim = sigma_input.shape
		sigma_input = sigma_input.view(-1, hidden_dim)         # Reshape for normalization
		sigma = self.sigma_predictor(sigma_input)
		sigma = sigma.view(batch_size, seq_len)                # Restore original shape
		sigma = 3*sigma + 1.0                                  # Map to [1.0, 4.0] range    [bs, m]
		
		c_input = c_input.view(-1, hidden_dim)
		c = self.c_predictor(c_input)
		c = c.view(batch_size, seq_len)
		c = c * self.c_scale
		
		b_input = b_input.view(-1, hidden_dim)
		b = self.b_predictor(b_input)
		b = b.view(batch_size, seq_len)
		b = b * self.b_scale

		sigma = sigma * mask        # mask [bs, m]
		c = c * mask
		b = b * mask

		return sigma, c, b, x1_fused, x2_fused


class ChannelSeparate1DCNN(nn.Module):
	"""Global correction factor from concatenated PL + LL probability sequences via 1D CNN."""
	def __init__(self, hidden_dim=128, dropout_rate=0.3):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.dropout_rate = dropout_rate
		
		self.conv_net = nn.Sequential(
			nn.Conv1d(1, hidden_dim//4, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(dropout_rate),
			nn.Conv1d(hidden_dim//4, hidden_dim//2, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Dropout(dropout_rate),
			nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Dropout(dropout_rate),
		)
		
		self.global_pool = nn.AdaptiveMaxPool1d(1)
		
		# Fully connected layers
		self.fc = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim//2),
			nn.ReLU(),
			nn.Dropout(dropout_rate),
			nn.Linear(hidden_dim//2, hidden_dim//4),
			nn.ReLU(),
			nn.Dropout(dropout_rate),
			nn.Linear(hidden_dim//4, 1),
		)
	
	def forward(self, probs, mask):
		"""
		Args:
			probs: [batch_size, seq_len] Probability values
			mask: [batch_size, seq_len] Mask
		Returns:
			output: [batch_size] Score for each complex
		"""
		# Ensure input validity
		assert probs.shape == mask.shape, "Probs and mask shape mismatch"
		
		# Set probs at invalid positions to 0
		masked_probs = probs * mask
		
		# Reshape for 1D convolution input format
		x = masked_probs.unsqueeze(1)
		
		# Apply convolution network
		x = self.conv_net(x)
		
		# Global pooling
		x = self.global_pool(x)
		x = x.squeeze(-1)
		
		# Fully connected layers
		x = self.fc(x)
		return x.squeeze(-1)


class LJModule(nn.Module):
	"""Lennard-Jones-style potential: E = (σ/D)¹² - 2(σ/D)⁶ per atom pair."""
	def __init__(self):
		super(LJModule, self).__init__()
		self.sigma_d_activation = SigmaOverDActivation()

	def forward(self, D, sigma, mask):
		sigma_over_D = torch.where(mask,
								sigma / (D + 1e-8), 
								torch.zeros_like(D)
		)
		sigma_over_D = self.sigma_d_activation(sigma_over_D)
		sigma_over_D6 = sigma_over_D ** 6
		sigma_over_D12 = sigma_over_D6 ** 2
		LJ = torch.where(mask,
						(sigma_over_D12) - 2*(sigma_over_D6),
						torch.zeros_like(sigma_over_D)
		)
		return LJ

class BoltzmannModule(nn.Module):
	"""Binding probability via Boltzmann-style formula: P = exp((-1 - E_i) / c)."""
	def __init__(self):
		super(BoltzmannModule, self).__init__()
		self.LJ = LJModule()
	
	def forward(self, D, sigma, c, mask):
		E_i = self.LJ(D, sigma, mask)
		E_j = -1.0

		delta_E = E_j - E_i
		c = torch.clamp(c, min=1e-8)
		prob = torch.where(mask,
							torch.exp(delta_E / c),
							torch.zeros_like(delta_E)
		)
		return prob, E_i

class SigmaOverDActivation(nn.Module):
	"""Clamp σ/D to [0, threshold] for numerical stability in (σ/D)¹² computation."""
	def __init__(self, threshold=1.2):
		super(SigmaOverDActivation, self).__init__()
		self.threshold = threshold

	def forward(self, x):
		x = torch.clamp(x, min=0.0)
		return torch.minimum(x, torch.tensor(self.threshold).to(x.device))


class APScore(nn.Module):
	def __init__(self, pl_acanet, 
						pl_atom_transformer,
						ll_acanet
					):
		super(APScore, self).__init__()
		
		# Protein-ligand network
		self.probability = BoltzmannModule()
		self.pl_acanet = pl_acanet
		self.aptransformer = pl_atom_transformer

		# Ligand-ligand network
		self.ll_acanet = ll_acanet
		
		# CNN
		self.cnn_compressor = ChannelSeparate1DCNN()

	def pad_and_stack(self, tensors, max_len=None, device=None):
		if max_len is None:
			max_len = max(t.size(0) for t in tensors)
			
		# Move all tensors to target device first
		tensors = [t.to(device) for t in tensors]
		
		padded = torch.stack([
			torch.cat([
				t.to(torch.float32),
				torch.zeros(max_len - t.size(0), *t.shape[1:], 
						dtype=torch.float32, device=device)
			]) for t in tensors
		])
		
		lengths = torch.tensor([t.size(0) for t in tensors], 
							device=device, dtype=torch.float32)
		return padded, lengths

	def get_max_length(self, *tensor_lists):
		"""Calculate maximum length across multiple tensor lists"""
		max_len = 0
		for tensors in tensor_lists:
			max_len = max(max_len, max(t.size(0) for t in tensors))
		return max_len

	def forward(self, batch):
		"""Compute binding score: ACANet → Boltzmann → Transformer → weighted sum → CNN correction."""
		device = next(self.parameters()).device

		# ── Stage 1: Pad all features to batch-uniform lengths ──
		pl_max_len = self.get_max_length(
			batch['pl_dist'],
			batch['prot_atom_feats'],
			batch['lig_atom_feats'],
			batch['prot_neighbor'],
			batch['lig_neighbor'],
			batch['angles'],
		)

		# Get ligand-ligand maximum length 
		ll_max_len = self.get_max_length(
			batch['ll_dist'],
			batch['ll_atom1_feats'], 
			batch['ll_atom2_feats'],
			batch['ll_neighbor1'],
			batch['ll_neighbor2'],
		)

		# ── Pad protein-ligand features ──
		D_dense, D_lengths = self.pad_and_stack(batch['pl_dist'], pl_max_len, device)
		prot_atom_dense, _ = self.pad_and_stack(batch['prot_atom_feats'], pl_max_len, device)
		lig_atom_dense, _ = self.pad_and_stack(batch['lig_atom_feats'], pl_max_len, device)
		prot_neighbor_dense, _ = self.pad_and_stack(batch['prot_neighbor'], pl_max_len, device)
		lig_neighbor_dense, _ = self.pad_and_stack(batch['lig_neighbor'], pl_max_len, device)
		angles_dense, _ = self.pad_and_stack(batch['angles'], pl_max_len, device)

		# ── Pad ligand-ligand features ──
		ll_D_dense, ll_D_lengths = self.pad_and_stack(batch['ll_dist'], ll_max_len, device)
		ll_atom1_dense, _ = self.pad_and_stack(batch['ll_atom1_feats'], ll_max_len, device)
		ll_atom2_dense, _ = self.pad_and_stack(batch['ll_atom2_feats'], ll_max_len, device)
		ll_neighbor1_dense, _ = self.pad_and_stack(batch['ll_neighbor1'], ll_max_len, device)
		ll_neighbor2_dense, _ = self.pad_and_stack(batch['ll_neighbor2'], ll_max_len, device)

		# ── Build masks for valid (non-padding) positions ──
		D_dense = D_dense.squeeze(-1)            # [bs, m, 1] -> [bs, m]
		pl_mask = torch.arange(pl_max_len, device=device)[None, :] < D_lengths[:, None]

		ll_D_dense = ll_D_dense.squeeze(-1)
		ll_mask = torch.arange(ll_max_len, device=device)[None, :] < ll_D_lengths[:, None]
		
		# ── Stage 2: ACANet → predict σ, c, b for protein-ligand pairs ──
		sigma, c, b, x1_fused, x2_fused = self.pl_acanet(prot_atom_dense,
														lig_atom_dense,
														prot_neighbor_dense, 
														lig_neighbor_dense,
														pl_mask)
		# -- Stage 3: Boltzmann probability for each atom pair --
		probs_ori, _ = self.probability(D_dense, sigma, c, pl_mask)
		pl_probs = probs_ori - b

		# -- Stage 4: AtomicPairTransformer -> learn importance weights --
		atom_pair_input = (x1_fused + x2_fused)
		w = self.aptransformer(atom_pair_input, sigma, angles_dense)
		w = w.squeeze(-1) * pl_mask

		# -- Stage 5: Weighted sum over protein-ligand pairs --
		pl_scores = pl_probs * w                                    # [bs, m]
		pl_scores = torch.sum(pl_scores * pl_mask, dim=1)               # [bs]

		# -- Stage 6: Ligand-ligand interaction scoring --
		sigma_lig, c_lig, b_lig, _, _ = self.ll_acanet(ll_atom1_dense,
													ll_atom2_dense,
													ll_neighbor1_dense,
													ll_neighbor2_dense,
													ll_mask)
		
		lig_probs, _ = self.probability(ll_D_dense, sigma_lig, c_lig, ll_mask)
		lig_probs = lig_probs-b_lig

		# -- Stage 7: CNN global correction factor from concatenated probs --
		probs = torch.cat([pl_probs, lig_probs], dim=1)            # [bs, m + n]
		mask = torch.cat([pl_mask, ll_mask], dim=1)

		cnn_score = self.cnn_compressor(probs, mask)
	
		final_scores = pl_scores * cnn_score                        # [bs]

		return  final_scores, D_dense, sigma, probs_ori, w

