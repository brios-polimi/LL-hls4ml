"""PyG HeteroData dataset over preprocessed .pt files."""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


class HeteroGraphDataset(Dataset):
    """
    Lazy-loading dataset for HeteroData .pt graphs.
    Indexes the filesystem at init time, loads graphs on demand.
    """

    def __init__(
        self,
        root: str | Path,
        types: list[str] | None = None,
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.paths = self._index(types)

    def _index(self, types: list[str] | None) -> list[Path]:
        paths = []
        root = self.root

        type_dirs = (
            [root / t for t in types]
            if types
            else [p for p in root.iterdir() if p.is_dir()]
        )

        for type_dir in sorted(type_dirs):
            if not type_dir.exists():
                raise FileNotFoundError(f"Type directory not found: {type_dir}")
            for archive_dir in sorted(type_dir.iterdir()):
                if not archive_dir.is_dir():
                    continue
                for pt_file in sorted(archive_dir.glob("*.pt")):
                    paths.append(pt_file)

        print(f"Indexed {len(paths)} graphs across {len(type_dirs)} type(s)")
        return paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> HeteroData:
        data = torch.load(self.paths[idx], weights_only=False)
        if self.transform:
            data = self.transform(data)
        return data

    def type_of(self, idx: int) -> str:
        """Return kernel type (e.g. 'exemplar') for a given index."""
        return self.paths[idx].parts[-3]
