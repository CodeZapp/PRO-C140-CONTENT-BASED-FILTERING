import pandas as pd 
import numpy as np 
df1 = pd.read_csv('sharedArticles.csv')
df2 = pd.read_csv('usersInteractions.csv')
print(df1.head())
print(df2.head())
df1 = df1[df1['eventType'] == 'CONTENT SHARED']
print(df1.head())
print(df2.shape)
print(df1.shape)
def findTotalEvents(df1Row):
    totalLikes = df2[(df2['contentId'] == df1Row['contentId']) & (df2['eventType'] == 'LIKE')].shape[0]
    totalViews = df2[(df2['contentId'] == df1Row['contentId']) & (df2['eventType'] == 'VIEW')].shape[0]
    totalBookmarks = df2[(df2['contentId'] == df1Row['contentId']) & (df2['eventType'] == 'BOOKMARK')].shape[0]
    totalFollows = df2[(df2['contentId'] == df1Row['contentId']) & (df2['eventType'] == 'FOLLOW')].shape[0]
    totalComments = df2[(df2['contentId'] == df1Row['contentId']) & (df2['eventType'] == 'COMMENT CREATED')].shape[0]
    return totalLikes + totalViews + totalBookmarks + totalFollows + totalComments
df1['totalEvents'] = df1.apply(findTotalEvents, axis = 1)
print(df1.head())
df1 = df1.sort_values(['totalEvents'], ascending = [False])
print(df1.head())
def convertLowercase(x):
    if isinstance(x, str):
        return x.lower()
    else:
        return ''
df1['title'] = df1['title'].apply(convertLowercase)
print(df1.head())
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words = 'english')
countMatrix = count.fit_transform(df1['title'])
from sklearn.metrics.pairwise import cosine_similarity
cosineSim2 = cosine_similarity(countMatrix, countMatrix)
df1 = df1.reset_index()
indices = pd.Series(df1.index, index = df1['contentId'])
def getRecommendations(contentId, cosineSim2):
    idx = indices[contentId]
    simScores = list(enumerate(cosineSim2[idx]))
    simScores = sorted(simScores, key=lambda x: x[1], reverse = True)
    simScores = simScores[1:11]
    movieIndices = [i[0] for i in simScores]
    return df1['contentId'].iloc[movieIndices]
getRecommendations(-4029704725707465084, cosineSim2)
df1.to_csv('articles.csv')