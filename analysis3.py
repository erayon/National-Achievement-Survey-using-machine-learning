
# coding: utf-8

# # Do students from South Indian states really excel at Math and Science?

# In[52]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
np.random.seed(5)
pd.set_option('chained_assignment',None)
get_ipython().magic('matplotlib notebook')


# In[53]:

df1 = pd.read_csv("./dataset/nas-columns.csv")


# In[54]:

df1.head(5)


# In[55]:

df2 = pd.read_csv('./dataset/nas-labels.csv')
df2[df2['Column']=="State"]


# In[56]:

df3 = pd.read_csv("./dataset/nas-pupil-marks.csv")


# In[57]:

df3.head(5)


# In[58]:

df3.columns


# In[59]:

dfx = df3[['STUID', 'State']]


# # remove 'STUID', 'State', 'District' from the Dataset

# In[60]:

nd = df3[['Gender', 'Age', 'Category',
       'Same language', 'Siblings', 'Handicap', 'Father edu', 'Mother edu',
       'Father occupation', 'Mother occupation', 'Below poverty',
       'Use calculator', 'Use computer', 'Use Internet', 'Use dictionary',
       'Read other books', '# Books', 'Distance', 'Computer use',
       'Library use', 'Like school', 'Subjects', 'Give Lang HW',
       'Give Math HW', 'Give Scie HW', 'Give SoSc HW', 'Correct Lang HW',
       'Correct Math HW', 'Correct Scie HW', 'Correct SocS HW',
       'Help in Study', 'Private tuition', 'English is difficult',
       'Read English', 'Dictionary to learn', 'Answer English WB',
       'Answer English aloud', 'Maths is difficult', 'Solve Maths',
       'Solve Maths in groups', 'Draw geometry', 'Explain answers',
       'SocSci is difficult', 'Historical excursions', 'Participate in SocSci',
       'Small groups in SocSci', 'Express SocSci views',
       'Science is difficult', 'Observe experiments', 'Conduct experiments',
       'Solve science problems', 'Express science views', 'Watch TV',
       'Read magazine', 'Read a book', 'Play games', 'Help in household',
       'Maths %', 'Reading %', 'Science %', 'Social %']]


# # Maping Categorical of 'use computer' and 'Subjects' to numerical value

# In[61]:

nd['Use computer'] = nd['Use computer'].map({"Yes":1,"No":0})
nd['Subjects'] = nd['Subjects'].map({'L':1, 'S':2, 'O':3, 'M':4, '0':0})


# # Preprosessing NaN value 

# In[62]:

from sklearn.preprocessing import Imputer


# In[63]:

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(nd)
tx = imp.transform(nd) 


# # Thresholding

# In[64]:

dc=['Gender', 'Age', 'Category',
       'Same language', 'Siblings', 'Handicap', 'Father edu', 'Mother edu',
       'Father occupation', 'Mother occupation', 'Below poverty',
       'Use calculator', 'Use computer', 'Use Internet', 'Use dictionary',
       'Read other books', '# Books', 'Distance', 'Computer use',
       'Library use', 'Like school', 'Subjects', 'Give Lang HW',
       'Give Math HW', 'Give Scie HW', 'Give SoSc HW', 'Correct Lang HW',
       'Correct Math HW', 'Correct Scie HW', 'Correct SocS HW',
       'Help in Study', 'Private tuition', 'English is difficult',
       'Read English', 'Dictionary to learn', 'Answer English WB',
       'Answer English aloud', 'Maths is difficult', 'Solve Maths',
       'Solve Maths in groups', 'Draw geometry', 'Explain answers',
       'SocSci is difficult', 'Historical excursions', 'Participate in SocSci',
       'Small groups in SocSci', 'Express SocSci views',
       'Science is difficult', 'Observe experiments', 'Conduct experiments',
       'Solve science problems', 'Express science views', 'Watch TV',
       'Read magazine', 'Read a book', 'Play games', 'Help in household',
       'Maths %', 'Reading %', 'Science %', 'Social %']
data = pd.DataFrame(tx,columns=dc)


# In[65]:

data.head(2)


# In[66]:

# Create a new column name stClass which defines as the best performance student
# as 1 : consider best performance
# as 0 : consider poor performance


# In[67]:

# Find the Performance by adding the Math, Science


# In[68]:

math    = np.array(data["Maths %"]).astype("float")
Science = np.array(data["Science %"]).astype("float")
performance = (math+Science)


# In[76]:

np.where(math==np.min(math))


# In[77]:

np.where(Science==np.min(Science))


# In[69]:

bestPerformance = np.max(performance)
poorPerformance = np.min(performance)
avgPerformance  = np.average(performance)


# In[70]:

poorPerformance


# In[19]:

# Find the Thresholing


# In[34]:

Threshold = bestPerformance-avgPerformance
super_threshold_indices = performance > Threshold
a = np.copy(performance)
a[super_threshold_indices] = 1
a[~super_threshold_indices]= 0


# In[35]:

# now check all student whos performance greater than 200 are consider as best


# In[36]:

performance = pd.DataFrame(performance,columns=["performance"])
Stclass     = pd.DataFrame(a,columns=["Stclass"])
state = pd.DataFrame(np.array(df3['State']),columns=["state"])


# In[37]:

modDf = pd.concat([state,data,performance,Stclass],axis=1)


# In[38]:

modDf.head(5)


# In[39]:

sNm=modDf.state.unique()
sNm


# In[40]:

southInd = ['AP','GA','KA','KL','PY','TN']


# In[41]:

def girlVsboy(modDf,state_name):
    state_           = modDf[modDf["state"]==state_name]
    state_boy        = state_[state_['Gender']==1].reset_index(drop=True)
    state_girl       = state_[state_['Gender']==2].reset_index(drop=True)
    performance_boy  = state_boy['Stclass']
    performance_girl = state_girl['Stclass']
    return (performance_girl,performance_boy)


# In[42]:

Total_nof_bestPrfmGr=[]
Total_nof_bestPrfmBy=[]
Total_nof_poorPrfmGr=[]
Total_nof_poorPrfmBy=[]

for i in southInd:
    tmp   = girlVsboy(modDf,i)
    bestG = np.sum(tmp[0])
    bestB = np.sum(tmp[1])
    poorG = len(tmp[0])-bestG
    poorB = len(tmp[1])-bestB
    Total_nof_bestPrfmGr.append(bestG)
    Total_nof_bestPrfmBy.append(bestB)
    Total_nof_poorPrfmGr.append(poorG)
    Total_nof_poorPrfmBy.append(poorB)


# In[43]:

import plotly.plotly as py
import plotly.graph_objs as go


# In[79]:

trace1 = go.Bar(
    x = southInd,
    y = Total_nof_bestPrfmGr,
    name='Girl'
)
trace2 = go.Bar(
    x = southInd,
    y = Total_nof_bestPrfmBy,
    name='Boy'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title='Best Performance of Girls and Boys South India',
    xaxis=dict(
        title='States',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='No of Student',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Best_Performance_of_Girls_and_Boys_South_India')


# In[81]:

from IPython.display import Image
Image(filename='./images/south_india_bestperfromance.png')


# In[80]:

trace1 = go.Bar(
    x = southInd,
    y = Total_nof_poorPrfmGr,
    name='Girl'
)
trace2 = go.Bar(
    x = southInd,
    y = Total_nof_poorPrfmBy,
    name='Boy'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title='Poor Performance of Girls and Boys South India',
    xaxis=dict(
        title='States',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='No of Student',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Poor_Performance_of_Girls_and_Boys_over_states')


# In[82]:

from IPython.display import Image
Image(filename='./images/south_india_poorPerformance.png')


# In[1]:

# yes


# In[ ]:



