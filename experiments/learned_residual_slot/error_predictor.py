"""Learned Residual SLOT: Train an error predictor during training for test-time correction.

Novel idea: during training, learn a small network E(context) → logit_correction
that predicts the base model's systematic errors. At eval time, apply this correction
BEFORE running L-BFGS SLOT, giving SLOT a better warm-start.

This is Bitter Lesson aligned:
- The correction is LEARNED from training data, not hand-designed
- It amortizes what L-BFGS would discover: systematic model biases
- More training data for E → better correction → better SLOT initialization

The error predictor is tiny (~50K params = 200KB in artifact) and trains
alongside the main model with zero impact on main model step time.

Usage:
    # During training: accumulate error statistics and train E
    predictor = ErrorPredictor(model_dim=512, vocab_size=1024)
    for step in training_loop:
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        # Every N steps, update error predictor (cheap, runs on CPU)
        if step % 10 == 0:
            with torch.no_grad():
                logits = model.forward_logits(x)
                predictor.update(logits, y, hidden_states=model.last_hidden)

    # At eval time: apply correction then SLOT
    correction = predictor.predict(hidden_avg)
    logits_corrected = logits + correction
    delta = lbfgs_slot(logits_corrected, ...)  # SLOT starts from better point
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ErrorPredictor(nn.Module):
    """Predicts systematic logit bias from context hidden state average.

    Architecture: LayerNorm → Linear(D, H) → GELU → Linear(H, V)
    where D = model_dim, H = hidden, V = vocab_size.

    The predictor learns: "given this context representation, what logit
    correction would minimize the model's errors?"

    At ~50K params for D=512, H=64, V=1024, this adds ~200KB to the artifact.
    """
    def __init__(self, model_dim: int, vocab_size: int, hidden_dim: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        # Initialize near-zero so initial correction is negligible
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # Running statistics for error accumulation
        self.register_buffer('error_sum', torch.zeros(vocab_size))
        self.register_buffer('error_count', torch.zeros(1))

    def forward(self, hidden_avg: Tensor) -> Tensor:
        """Predict logit correction from averaged hidden states.

        Args:
            hidden_avg: (batch, model_dim) — average hidden state of context
        Returns:
            correction: (batch, vocab_size) — predicted logit bias
        """
        x = self.norm(hidden_avg)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

    @torch.no_grad()
    def accumulate_errors(self, logits: Tensor, targets: Tensor):
        """Accumulate model prediction errors for offline analysis.

        Args:
            logits: (batch, seq, vocab) — model's logit predictions
            targets: (batch, seq) — ground truth token IDs
        """
        # Compute per-vocab error: how much does the model under/over-predict each token?
        probs = F.softmax(logits.float(), dim=-1)
        batch, seq, vocab = probs.shape
        # One-hot targets
        targets_flat = targets.reshape(-1)
        one_hot = torch.zeros(batch * seq, vocab, device=logits.device)
        one_hot.scatter_(1, targets_flat.unsqueeze(1), 1.0)
        # Error: how much we under-predict each token (positive = need to boost)
        error = (one_hot - probs.reshape(-1, vocab)).mean(dim=0)  # (vocab,)
        self.error_sum += error.to(self.error_sum.device)
        self.error_count += 1

    def get_average_error(self) -> Tensor:
        """Get the average prediction error across accumulated batches."""
        if self.error_count.item() == 0:
            return self.error_sum
        return self.error_sum / self.error_count

    def get_baseline_correction(self) -> Tensor:
        """Get a static logit correction based on accumulated errors.

        This is the simplest "learned" correction: just the average error.
        The neural predictor can do better by conditioning on context.
        """
        avg_error = self.get_average_error()
        # Scale: the correction should be in logit space
        # avg_error is in probability space, convert roughly via log
        # correction ≈ log(1 + error/prob) ≈ error/prob for small errors
        # Simplified: just use the error directly, clamped
        return avg_error.clamp(-5.0, 5.0)


class ErrorPredictorTrainer:
    """Trains the error predictor alongside the main model.

    The trainer collects (hidden_avg, optimal_correction) pairs during
    training and fits the predictor to predict corrections from context.
    """
    def __init__(self, predictor: ErrorPredictor, lr: float = 1e-3):
        self.predictor = predictor
        self.optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
        self.buffer_hidden = []
        self.buffer_correction = []
        self.buffer_max = 1000

    @torch.no_grad()
    def collect_sample(self, hidden_states: Tensor, logits: Tensor, targets: Tensor):
        """Collect a (hidden_avg, optimal_correction) sample.

        Args:
            hidden_states: (batch, seq, dim) — model's hidden states
            logits: (batch, seq, vocab) — model's logit predictions
            targets: (batch, seq) — ground truth
        """
        # Average hidden state across sequence
        hidden_avg = hidden_states.mean(dim=1).float()  # (batch, dim)

        # Compute "optimal" correction: what logit bias minimizes CE loss?
        # For a single batch, this is approximately:
        # correction_v = log(count_v / expected_v) for each vocab token v
        batch, seq, vocab = logits.shape
        probs = F.softmax(logits.float(), dim=-1).mean(dim=(0, 1))  # (vocab,) avg prediction
        targets_flat = targets.reshape(-1)
        counts = torch.bincount(targets_flat, minlength=vocab).float()
        counts = counts / counts.sum()  # normalize
        # Correction: boost under-predicted tokens, suppress over-predicted
        correction = (counts / (probs + 1e-8)).log().clamp(-5.0, 5.0)  # (vocab,)

        # Store
        for i in range(min(hidden_avg.shape[0], 4)):  # subsample to avoid memory blowup
            self.buffer_hidden.append(hidden_avg[i:i+1].cpu())
            self.buffer_correction.append(correction.unsqueeze(0).cpu())

        # Trim buffer
        if len(self.buffer_hidden) > self.buffer_max:
            self.buffer_hidden = self.buffer_hidden[-self.buffer_max:]
            self.buffer_correction = self.buffer_correction[-self.buffer_max:]

    def train_step(self):
        """Train the predictor on collected samples."""
        if len(self.buffer_hidden) < 10:
            return 0.0

        device = next(self.predictor.parameters()).device
        # Sample a mini-batch
        n = min(len(self.buffer_hidden), 32)
        indices = torch.randperm(len(self.buffer_hidden))[:n]
        hidden_batch = torch.cat([self.buffer_hidden[i] for i in indices]).to(device)
        correction_batch = torch.cat([self.buffer_correction[i] for i in indices]).to(device)

        # Forward + loss
        pred = self.predictor(hidden_batch)
        loss = F.mse_loss(pred, correction_batch)

        # Backward + step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


if __name__ == "__main__":
    # Quick test
    model_dim, vocab_size = 512, 1024
    predictor = ErrorPredictor(model_dim, vocab_size, hidden_dim=64)
    print(f"ErrorPredictor params: {sum(p.numel() for p in predictor.parameters()):,}")
    print(f"Size at fp16: {sum(p.numel() for p in predictor.parameters()) * 2 / 1024:.1f} KB")
    print(f"Size at int8: {sum(p.numel() for p in predictor.parameters()) / 1024:.1f} KB")

    # Test forward
    hidden_avg = torch.randn(4, model_dim)
    correction = predictor(hidden_avg)
    print(f"Input: {hidden_avg.shape}, Output: {correction.shape}")
    print(f"Output range: [{correction.min().item():.4f}, {correction.max().item():.4f}]")
