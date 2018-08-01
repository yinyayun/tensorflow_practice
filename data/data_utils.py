'''
Created on 2018年7月31日

@author: yinyayun
'''
import zipfile

TAGS_ID = {
'news_culture':0,
'news_sports':1,
'news_entertainment':2,
'news_finance':3,
'news_house':4,
'news_car':5,
'news_edu':6,
'news_tech':7,
'news_military':8,
'news_travel':9,
'news_world':10,
'news_agriculture':11,
'news_game':12,
'stock':13,
'news_story':14,
}

def loadZip(path='toutiao_cat_data_tokens.zip', filters=['news_culture', 'news_sports']):
    inputs = []
    targets = []
    with zipfile.ZipFile(path) as zipf:
        for name in zipf.namelist():
            with zipf.open(name) as f:
                for line in f:
                    parts = line.decode('utf-8')[:-2].split('_!_')
                    if(filterTag(filters, parts[0])):
                        inputs.append(parts[1])
                        targets.append(TAGS_ID[parts[0]])
    return inputs, targets

def filterTag(filters, tag):
    if(len(filters) == 0):
        return True
    for name in filters:
        if(tag == name):
            return True
    return False

if __name__ == '__main__':
    loadZip('toutiao_cat_data_tokens.zip', ['news_culture', 'news_sports'])
