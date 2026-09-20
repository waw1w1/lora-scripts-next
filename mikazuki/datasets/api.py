from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mikazuki.app.models import APIResponseSuccess
from mikazuki.datasets.listing import list_datasets
from mikazuki.datasets.root import (
    DEFAULT_DATASETS_ROOT,
    get_datasets_root,
    normalize_path,
    set_datasets_root,
)
from mikazuki.datasets.sandbox import resolve_dataset_dir

router = APIRouter()


class RootUpdateRequest(BaseModel):
    path: str


class DatasetCreateRequest(BaseModel):
    name: str


def root_payload() -> dict:
    root = get_datasets_root()
    return {
        "root": normalize_path(root),
        "default": DEFAULT_DATASETS_ROOT,
        "exists": root.is_dir(),
    }


@router.get("/datasets/root")
async def get_root():
    return APIResponseSuccess(data=root_payload())


@router.put("/datasets/root")
async def update_root(req: RootUpdateRequest):
    try:
        set_datasets_root(req.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"cannot create datasets root: {exc}") from exc
    return APIResponseSuccess(data=root_payload())


@router.get("/datasets")
async def list_all():
    root = get_datasets_root()
    return APIResponseSuccess(
        data={
            "root": normalize_path(root),
            "exists": root.is_dir(),
            "datasets": list_datasets(root),
        }
    )


@router.post("/datasets")
async def create(req: DatasetCreateRequest):
    root = get_datasets_root()
    dataset_dir = resolve_dataset_dir(root, req.name)
    if dataset_dir.exists():
        raise HTTPException(status_code=409, detail="dataset already exists")
    try:
        dataset_dir.mkdir(parents=True)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"cannot create dataset: {exc}") from exc
    return APIResponseSuccess(data={"name": dataset_dir.name, "path": normalize_path(dataset_dir)})
