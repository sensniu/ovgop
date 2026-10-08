from torch.utils.tensorboard import SummaryWriter
import os


class TensorboardLogger:
    """Manage one shared TensorBoard writer."""

    writer = None

    @classmethod
    def initialize(cls, log_dir: str):
        """Create the writer on first use."""
        if cls.writer is None:
            os.makedirs(log_dir, exist_ok=True)
            cls.writer = SummaryWriter(log_dir)
            print(f"TensorBoard Logger initialized to: {log_dir}")
        else:
            print("Warning: TensorBoard Logger already initialized.")

    @classmethod
    def get_writer(cls) -> SummaryWriter:
        """Return the initialized writer."""
        if cls.writer is None:
            raise RuntimeError("TensorBoard Logger not initialized. Call initialize() first.")
        return cls.writer

    @classmethod
    def close(cls):
        """Close the writer."""
        if cls.writer:
            cls.writer.close()
            cls.writer = None
