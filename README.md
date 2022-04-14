# hm
[hm kaggle competition](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/overview)


```

.
├── docs
│   ├── articles.csv
│   ├── customers.csv
│   ├── data.csv
│   ├── data.zip
│   ├── head_transactions_train.csv
│   ├── sample_submission.csv
│   └── transactions_train.csv
├── LICENSE
├── README.md
├── batch.py
├── lstm.py
├── main.py
├── preprocess.py
└── requirements.txt

```

### NOTES:

1. installed git
    - `sudo apt install git`
2. downloaded data from kaggle    

### TODO:

- add requirements.txt

- kmeans
    - clients

- 4 layer mlp
    - predict 1 product
    - predict 12 products
    - custom output layer / loss function
        - randomly picks product based on distribution from softmax weights

- cnn
    - images
