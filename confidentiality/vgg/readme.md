# Steps for training 

Directly testing the stealing attack results by:

```shell
cd figure7/7a/
conda activate teeinfer
python3 test.py
```

**Notes:** To run ``test.py``, you will need the ``*.pkl`` in ``figure7/data/``.

If you want to train each data point by yourself, you can run the following steps:

```shell
cd figure7/7a/
conda activate teeinfer
python3 train.py
```

The training will take around 2-3 hours.
