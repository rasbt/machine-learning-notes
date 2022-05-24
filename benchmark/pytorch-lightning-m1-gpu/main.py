import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import time
import torch
from torchvision import transforms
from watermark import watermark

from my_classifier_template.dataset import Cifar10DataModule
from my_classifier_template.model import LightningClassifier


def parse_cmdline_args(parser=None):

    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--accelerator',
                        type=str,
                        default="auto")

    parser.add_argument('--batch_size',
                        type=int,
                        default=32)

    parser.add_argument('--data_path',
                        type=str,
                        default='./data')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0005)

    parser.add_argument('--log_accuracy',
                        type=str,
                        choices=("true", "false"),
                        default="true")

    parser.add_argument('--mixed_precision',
                        type=str,
                        choices=("true", "false"),
                        default="true")

    parser.add_argument('--num_epochs',
                        type=int,
                        default=10)

    parser.add_argument('--num_workers',
                        type=int,
                        default=3)

    parser.add_argument('--output_path',
                        type=str,
                        required=True)

    parser.add_argument('--pretrained',
                        type=str,
                        choices=("true", "false"),
                        default="false")

    parser.add_argument('--num_devices',
                        nargs="+",
                        default="auto")

    parser.add_argument('--device_numbers',
                        type=str,
                        default="")

    parser.add_argument('--random_seed',
                        type=int,
                        default=-1)
                        
    parser.add_argument('--strategy',
                        type=str,
                        default="")

    parser.set_defaults(feature=True)
    args = parser.parse_args()

    if not args.strategy:
        args.strategy = None

    if args.num_devices != "auto":
        args.devices = int(args.num_devices[0])
    if args.device_numbers:
        args.devices = [int(i) for i in args.device_numbers.split(',')]

    d = {'true': True,
         'false': False}

    args.log_accuracy = d[args.log_accuracy]
    args.pretrained = d[args.pretrained]
    args.mixed_precision = d[args.mixed_precision]
    if args.mixed_precision:
        args.mixed_precision = 16
    else:
        args.mixed_precision = 32

    return args


if __name__ == "__main__":

    print(watermark())
    print(watermark(packages="torch,pytorch_lightning"))

    parser = argparse.ArgumentParser()
    args = parse_cmdline_args(parser)

    torch.manual_seed(args.random_seed)

    custom_train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )

    custom_test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )

    data_module = Cifar10DataModule(
        batch_size=args.batch_size,
        data_path=args.data_path,
        num_workers=args.num_workers,
        train_transform=custom_train_transform,
        test_transform=custom_test_transform)

    pytorch_model = torch.hub.load(
        'pytorch/vision:v0.11.0',
        'mobilenet_v3_large',
        pretrained=args.pretrained)

    pytorch_model.classifier[-1] = torch.nn.Linear(
        in_features=1280, out_features=10  # as in original
    )  # number of class labels in Cifar-10)

    lightning_model = LightningClassifier(
        pytorch_model, learning_rate=args.learning_rate, log_accuracy=args.log_accuracy)

    if args.log_accuracy:
        callbacks = [
            ModelCheckpoint(
                save_top_k=1, mode="max", monitor="valid_acc"
            )  # save top 1 model
        ]
    else:
        callbacks = [
            ModelCheckpoint(
                save_top_k=1, mode="min", monitor="valid_loss"
            )  # save top 1 model
        ]

    logger = CSVLogger(save_dir=args.output_path, name="my-model")

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        strategy=args.strategy,
        precision=args.mixed_precision,
        deterministic=False,
        log_every_n_steps=10,
    )

    start_time = time.time()
    trainer.fit(model=lightning_model, datamodule=data_module)

    train_time = time.time()
    runtime = (train_time - start_time) / 60
    print(f"Training took {runtime:.2f} min.")

    before = time.time()
    val_acc = trainer.test(dataloaders=data_module.val_dataloader())
    runtime = (time.time() - before) / 60
    print(f"Inference on the validation set took {runtime:.2f} min.")

    before = time.time()
    test_acc = trainer.test(dataloaders=data_module.test_dataloader())
    runtime = (time.time() - before) / 60
    print(f"Inference on the test set took {runtime:.2f} min.")

    runtime = (time.time() - start_time) / 60
    print(f"The total runtime was {runtime:.2f} min.")

    print("Validation accuracy:", val_acc)
    print("Test accuracy:", test_acc)
