# train
python tools/run.py fit --config configs/cityscapes_darkzurich/ICDA_daformer.yaml --trainer.gpus [0] --trainer.precision 16
# val
# python tools/run.py validate --config configs/cityscapes_darkzurich/ICDA_daformer.yaml --ckpt_path /path/to/trained/model --trainer.gpus [0]
# test
# python tools/run.py test --config configs/cityscapes_darkzurich/ICDA_daformer.yaml --ckpt_path /path/to/trained/model --trainer.gpus [0]
# predict
# python tools/run.py predict --config configs/cityscapes_darkzurich/ICDA_daformer.yaml --ckpt_path /path/to/trained/model --trainer.gpus [0]