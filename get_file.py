import requests

url = "http://cxq-ns1:50070/webhdfs/v1/srv/multi-tenant/midteant01/dev/user/p14_bupt/008/etl_output/train_data.csv?op=OPEN&user.name=zhanghengji"

r = requests.get(url, stream=True)

with open("train_data.csv", "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
