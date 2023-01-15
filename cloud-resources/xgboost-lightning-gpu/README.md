# Training an XGBoost Classifier Using Cloud GPUs Without Worrying About Infrastructure



Code accompanying the blog article: [Training an XGBoost Classifier Using Cloud GPUs Without Worrying About Infrastructure](https://sebastianraschka.com/blog/2023/xgboost-gpu.html).



Run code as follows:



```pip install lightning
# run XGBoost classifier locally
python my_xgboost_classifier.py 

# run XGBoost classifier locally via Lightning (if you have a GPU)
pip install lightning
lightning run app xgboost-cloud-gpu.py --setup

# run XGBoost in Lightning cloud on a V100
lightning run app xgboost-cloud-gpu.py --cloud
```

