import lightning as L
import torch


class Callback:
    """Base class for callbacks."""

    def on_fit_start(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102
        pass

    def on_fit_end(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102
        pass

    def on_after_setup(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102
        pass

    def on_train_start(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102
        pass

    def on_validation_start(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102
        pass

    def on_test_start(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102
        pass

    def on_train_end(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102
        pass

    def on_validation_end(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102
        pass

    def on_test_end(self, *, fabric: L.Fabric, module: torch.nn.Module) -> None:  # noqa: D102
        pass

    def on_train_batch_start(  # noqa: D102
        self,
        *,
        fabric: L.Fabric,
        module: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        pass

    def on_validation_batch_start(  # noqa: D102
        self, *, fabric: L.Fabric, module: torch.nn.Module, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        pass

    def on_test_batch_start(  # noqa: D102
        self, *, fabric: L.Fabric, module: torch.nn.Module, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        pass

    def on_train_batch_end(  # noqa: D102
        self,
        *,
        fabric: L.Fabric,
        module: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        pass

    def on_validation_batch_end(  # noqa: D102
        self,
        *,
        fabric: L.Fabric,
        module: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        pass

    def on_test_batch_end(  # noqa: D102
        self,
        *,
        fabric: L.Fabric,
        module: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        pass
