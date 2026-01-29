"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
import torch.nn.functional as F


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.layers = nn.ModuleList()
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.output_layer = nn.Linear(input_dim, chunk_size * action_dim)
        self.activation = nn.ReLU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_action_chunk = self.forward(state)
        loss = F.mse_loss(pred_action_chunk, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            action_chunk = self.forward(state)
        return action_chunk


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # Input: state + flattened action chunk + time step
        input_dim = state_dim + chunk_size * action_dim + 1
        self.layers = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.output_layer = nn.Linear(input_dim, chunk_size * action_dim)
        self.activation = nn.ReLU()

    def forward(
            self,
            state: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field given state, noisy action x_t, and time t.
        
        Args:
            state: [batch, state_dim]
            x_t: [batch, chunk_size, action_dim] - interpolated action at time t
            t: [batch] or [batch, 1] - time step in [0, 1]
        
        Returns:
            velocity: [batch, chunk_size, action_dim] - predicted velocity field
        """
        batch_size = state.shape[0]
        t = t.view(batch_size, 1)
        x = torch.cat([state, x_t.view(batch_size, -1), t], dim=-1)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flow matching loss.
        
        Args:
            state: [batch, state_dim]
            action_chunk: [batch, chunk_size, action_dim] - target action (x_1)
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Sample time t ~ U(0, 1)
        t = torch.rand(batch_size, 1, 1, device=device)
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(action_chunk)
        
        # Interpolate: x_t = t * x_1 + (1 - t) * x_0
        x_t = t * action_chunk + (1 - t) * x_0
        
        # Predict velocity field
        pred_velocity = self.forward(state, x_t, t.squeeze(-1))
        
        # Target velocity: v* = x_1 - x_0
        target_velocity = action_chunk - x_0
        
        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Generate actions by integrating the learned velocity field.
        
        Args:
            state: [batch, state_dim]
            num_steps: number of Euler integration steps
        
        Returns:
            action_chunk: [batch, chunk_size, action_dim]
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Start from noise x_0 ~ N(0, I)
        x = torch.randn(batch_size, self.chunk_size, self.action_dim, device=device)
        
        # Euler integration from t=0 to t=1
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            with torch.no_grad():
                velocity = self.forward(state, x, t)
            x = x + velocity * dt
        
        return x
        


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
