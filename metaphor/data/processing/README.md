The script in `process_pheme15` converts the extended Pheme ![dataset](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078).

For each source tweet, store: tweet id, timestamp, topic, text and veracity in a csv.

Run as 
`python process_pheme.py "/Users/yousuf/Documents/ucl-msc/meta-misinformation-detection/data/pheme/all-rnr-annotated-threads" "/Users/yousuf/Documents/ucl-msc/meta-misinformation-detection/data/pheme/processed-pheme.csv"`.