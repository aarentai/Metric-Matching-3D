import pandas as pd
import requests
date_list = pd.date_range(start='2019-01-01',end='2019-12-31')
for i in range(len(date_list)):
    dateforurl = date_list[i].replace('-','/')
    url = 'https://www.nytimes.com/images/'+dateforurl+'/nytfrontpage/scan.pdf'
    r = requests.get(url, stream=True)

    dateforfile = date_list[i].replace('-','')
    with open('E:/Temporary/NYTimes/NYT'+dateforfile+'.pdf', 'wb') as f:
        f.write(r.content)
