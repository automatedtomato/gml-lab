import pytest

from src.gml_lab.config_builder import build_config

model_archs = [
    "resnet18_8xb32_in1k",
    "vit-base-p16_64xb64_in1k",
]
data_setting = "imagenet_lmdb"


@pytest.mark.parametrize("arch", model_archs)
def test_build_config(arch: str) -> None:
    cfg = build_config(
        model_arch=arch,
        data_setting=data_setting,
    )

    assert cfg.val_dataloader.dataset.type == "ImageNetLMDB"
