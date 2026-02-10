custom_imports = dict(imports=["src.gml_lab.dataset"], allow_failed_imports=False)

val_pipeline = [
    dict(type="LoadImageFromLMDB"),
    dict(type="Resize", scale=256, keep_ratio=True),
    dict(type="CenterCrop", crop_size=224),
    dict(type="PackInputs"),
]

test_pipeline = val_pipeline

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ImageNetLMDB",
        lmdb_path="data/imagenet/val_lmdb",
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="Accuracy", topk=(1, 5))
test_evaluator = val_evaluator
