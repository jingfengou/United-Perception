from abc import ABC, abstractmethod


class TrainingBackend(ABC):
    @abstractmethod
    def autocast(self, enabled=True, dtype=None, cache_enabled=True):
        """Return a context manager. Do not change persistent precision defaults."""

    @abstractmethod
    def grad_scaler(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
        """Return a scaler supporting step/update and original state_dict keys."""

    @abstractmethod
    def wrap_ddp(self, module, **kwargs):
        """Return distributed wrapper, passing original DDP kwargs unchanged."""

    @abstractmethod
    def wrap_fsdp(self, module, *, shard_modules=(), reshard_after_forward=True):
        """FSDP2 in-place sharding on the vendor device and default process group.

        Requires PyTorch 2.10+ FSDP2/DTensor and an initialized multi-rank group.
        Broadcast initial parameters/buffers from rank 0, shard named submodules
        bottom-up and then the root. Names/shapes and forward/autograd stay intact;
        construct the optimizer afterwards. No implicit precision conversion.
        """

    @abstractmethod
    def sharded_grad_scaler(self, init_scale=65536.0, growth_factor=2.0,
                            backoff_factor=0.5, growth_interval=2000, enabled=True):
        """AMP scaler for FSDP2 gradients; synchronize overflow across all ranks.

        Unscale GPU-local DTensor shards in place; a nonfinite shard on any rank
        skips the update on every rank. Preserve ordinary GradScaler state keys.
        """

    @abstractmethod
    def full_state_dict(self, module, optimizer=None):
        """Collect canonical full model/optional optimizer state on rank 0 CPU.

        All ranks must call, including for EMA. Returns a dict with model and
        optional optimizer keys; optimizer parameters use canonical model names.
        Other ranks return empty state dicts. No persistent model/optimizer update.
        """

    @abstractmethod
    def load_full_state_dict(self, module, state, optimizer=None, *, strict=True):
        """Load full model and/or optimizer state on every rank before backward.

        All ranks supply the same CPU checkpoint. Shard through vendor DTensor;
        support canonical named optimizer states and legacy ordered parameter IDs.
        Model/optimizer keys may be omitted independently. Preserve parameter
        groups, names and shapes; reject incompatible groups rather than reset.
        """

    @abstractmethod
    def validation(self, module):
        """eval/no_grad context; restore every submodule's entry mode on success/error."""

    def fused_optimizer(self, name, params, **kwargs):
        """Optional legacy extension, outside the required backend contract."""
        from up_platform.contracts import UnsupportedOperation
        raise UnsupportedOperation(self.device.vendor, 'training', name,
                                   'Optional legacy optimizer is not enabled')

    @abstractmethod
    def optimizer(self, name, **kwargs):
        """Construct the original torch.optim class with original parameter groups."""

    @abstractmethod
    def clip_grad_norm_(self, parameters, max_norm, norm_type=2.0, **kwargs):
        """Original in-place gradient clipping and returned total norm."""
