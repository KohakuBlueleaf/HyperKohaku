import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from diffusers.models.attention import Attention
from diffusers.loaders import AttnProcessor

try:
    import xformers.ops
except:
    pass

class LiLoRALinearLayer(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        down_dim=100, 
        up_dim=50, 
        rank=1, 
        network_alpha=None, 
        trained=False
    ):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")

        self.register_buffer('aux_seed', torch.randint(0, 2**32-1, (1,)))
        self.in_features = in_features
        self.out_features = out_features
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.down = self.up = None
        self.split = (down_dim*rank, up_dim*rank)
        self.network_alpha = network_alpha
        self.rank = rank
        self.scale = network_alpha / rank if network_alpha is not None else 1.0
        self.trained = trained
        if trained:
            self.weight = nn.Parameter(torch.empty(down_dim+up_dim, dtype=torch.float32))

    def make_weight(self):
        assert self.down is not None and self.up is not None
        assert (self.down.dim() == 2 or self.down.size(0) == 1
                and self.up.dim() == 2 or self.up.size(0) == 1)
        seed = self.aux_seed.item()
        del self.aux_seed
        self.aux_seed = seed
        with torch.no_grad():
            down = self.down.reshape(-1, self.in_features)
            self.down = nn.Linear(self.in_features, self.rank, bias=False)
            self.down.weight = nn.Parameter(down)
            up = self.up.reshape(self.out_features, -1)
            self.up = nn.Linear(self.rank, self.out_features, bias=False)
            self.up.weight = nn.Parameter(up)

    def update_weight(self, weight: torch.Tensor, add_constant=False):
        '''
        weight: [b, up_dim+down_dim] or [up_dim+down_dim]
        '''
        # get aux weights
        down_aux = torch.empty(self.down_dim, self.in_features, dtype=weight.dtype, device=weight.device)
        up_aux = torch.empty(self.out_features, self.up_dim, dtype=weight.dtype, device=weight.device)
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.aux_seed.item())
        nn.init.orthogonal_(down_aux, gain=1)
        nn.init.orthogonal_(up_aux, gain=1)
        torch.random.set_rng_state(rng_state)
        
        down, up = weight.split(self.split, dim=-1)
        if weight.dim() == 1:
            down = down.reshape(self.rank, -1)
            up = up.reshape(-1, self.rank)
        elif weight.dim() == 2:
            down = down.reshape(weight.size(0), self.rank, -1)
            up = up.reshape(weight.size(0), -1, self.rank)
        else:
            raise ValueError(f"weight dim {weight.dim()} is not supported")
        
        if add_constant:
            down = down + 1
        self.down = down @ down_aux             #[..., rank, in]
        self.up = up_aux @ up                   #[..., out, rank]

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.dtype
        if self.trained:
            self.update_weight(self.weight)

        if self.down.dim()==3:
            down_hidden_states = torch.einsum('b o i, b ... i -> b ... o', self.down, hidden_states.to(dtype))
            up_hidden_states = torch.einsum('b o i, b ... i -> b ... o', self.up, down_hidden_states)
        else:
            down_hidden_states = F.linear(hidden_states.to(dtype), self.down)
            up_hidden_states = F.linear(down_hidden_states, self.up)

        if self.network_alpha is not None:
            up_hidden_states *= self.scale

        return up_hidden_states.to(orig_dtype)


class LiLoRAAttnProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, down_dim=100, up_dim=50, rank=4, network_alpha=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        self.to_q_lora = LiLoRALinearLayer(hidden_size, hidden_size, down_dim, up_dim, rank, network_alpha)
        self.to_k_lora = LiLoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, down_dim, up_dim, rank, network_alpha)
        self.to_v_lora = LiLoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, down_dim, up_dim, rank, network_alpha)
        self.to_out_lora = LiLoRALinearLayer(hidden_size, hidden_size, down_dim, up_dim, rank, network_alpha)

    @property
    def layers(self):
        return self.to_q_lora, self.to_k_lora, self.to_v_lora, self.to_out_lora

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LiLoRAXformersAttnProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, down_dim=100, up_dim=50, rank=4, attention_op = None, network_alpha=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.attention_op = attention_op

        self.to_q_lora = LiLoRALinearLayer(hidden_size, hidden_size, down_dim, up_dim, rank, network_alpha)
        self.to_k_lora = LiLoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, down_dim, up_dim, rank, network_alpha)
        self.to_v_lora = LiLoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, down_dim, up_dim, rank, network_alpha)
        self.to_out_lora = LiLoRALinearLayer(hidden_size, hidden_size, down_dim, up_dim, rank, network_alpha)

    @property
    def layers(self):
        return self.to_q_lora, self.to_k_lora, self.to_v_lora, self.to_out_lora

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # hidden_states = torch.bmm(attention_probs, value)
        hidden_states = xformers.ops.memory_efficient_attention(
            query.to(key.dtype), key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states