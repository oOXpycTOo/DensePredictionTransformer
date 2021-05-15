import argparse
import importlib

import pytorch_lightning as pl
import wandb

from detectors import lit_models
from detectors.callbacks.wandb_callbacks import ImageCallback


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.ViT'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Setup Python argparse parser with data, model, trainer and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = 'Trainer Args'

    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument('--data_class', type=str, default='GlassSegmentationDataModule')
    parser.add_argument('--model_class', type=str, default='DensePredictionTransformer')
    parser.add_argument('--load_checkpoint', type=str, default=None)

    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f'detectors.data.{temp_args.data_class}')
    model_class = _import_class(f'detectors.models.{temp_args.model_class}')

    data_group = parser.add_argument_group('Data Args')
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group('Model Args')
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group('LitModel Args')
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument('--help', '-h', action='help')
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f'detectors.data.{args.data_class}')
    model_class = _import_class(f'detectors.models.{args.model_class}')
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)
    lit_model_class = lit_models.BaseLitModel
    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)
    logger = pl.loggers.TensorBoardLogger('training/logs')
    if args.wandb:
        logger = pl.loggers.WandbLogger()
        logger.watch(model)
        logger.log_hyperparams(vars(args))

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='{epoch:03d}-{val_loss:.3f}-{val_mAP:.3f}', monitor='val_loss', mode='min'
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]
    if args.wandb:
        image_callback = ImageCallback(args)
        callbacks.append(image_callback)
    # args.weights_summary = 'full'
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, weights_save_path='training/logs')

    trainer.tune(lit_model, datamodule=data)

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)

    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        print('Best model saved at: ', best_model_path)
        if args.wandb:
            wandb.save(best_model_path)

if __name__ == '__main__':
    main()
