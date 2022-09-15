Check usage via:

```
python mlp_cli2.py --help
```

```
<class 'shared_utilities.LightningModel'>:
  --model CONFIG        Path to a configuration file.
  --model.model MODEL   (type: Optional[Any], default: null)
  --model.learning_rate LEARNING_RATE
                        (type: Optional[Any], default: null)

<class 'shared_utilities.CustomDataModule'>:
  --data CONFIG         Path to a configuration file.
  --data.data_dir DATA_DIR
                        (type: Any, default: ./mnist)
  --data.batch_size BATCH_SIZE
                        (type: Any, default: 64)
```



Usage example:

```
python mlp_cli2.py --model.learning_rate 0.1 --model.hidden_units "[100, 200]"
```
