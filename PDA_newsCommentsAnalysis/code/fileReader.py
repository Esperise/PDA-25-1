import os
import pandas as pd
import re

from dask.dataframe import to_csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target_path = os.path.join(BASE_DIR, "src", "news_comments")
# base_path= "../src/news_comments"
folders=[
    name for name in os.listdir(target_path)
    if os.path.isdir(os.path.join(target_path))
]
print(folders)
df_all=pd.DataFrame()
count=0
for folder in folders:
    path=f"../{target_path}/{folder}"
    try:
        in_folder= [
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path))
        ]
        print(in_folder)
        print(f"{folder}: {len(in_folder)}")
    except NotADirectoryError as e:
        continue
    for csv_name in in_folder:
        count+=1
        try:
            date=re.findall(r"\d{4}-\d{2}-\d{2}",csv_name)[0]
            df_read = pd.read_csv(f"{path}/{csv_name}")
            df_read.insert(0, "date", date)
            df_read.insert(0, "press_name", folder)
            # article_name=re.split(r"\d{4}-\d{2}-\d{2}_|\.csv", csv_name)[1].split(f"{folder}_")[0]
            # print(article_name)
            # df_read.insert(0,"article_name",article_name)
            df_all = pd.concat([df_all, df_read])
            print(f"{count} \n {folder}-{csv_name} 통합 완료 ")
            with open("./log_integrating_csv4_all.txt", "a") as file:
                file.write(f"{count}{in_folder}-{csv_name} 통합 완료 \n")
        except IndexError as e:
            print(e)

df_all.to_csv(f"../src/integrated_csv/integrated_csv.csv",index=False)

