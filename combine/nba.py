#imports the modules that I will need
import pandas as pd
import numpy as np
import math
import time


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

from matplotlib import pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

#(csv 1)this dataset is the main one it provides data on shots and goals and what type of shot was used
df = pd.read_csv('combined.csv')
df_team = pd.read_csv('teamdstats.csv')

print(df)

df.drop(columns='Unnamed: 0', inplace=True)
df.drop(columns='Unnamed: 0.1', inplace=True)
df.drop(columns='Unnamed: 0.2', inplace=True)
df.drop(columns='match_id', inplace=True)
df.drop(columns='status', inplace=True)
df.drop(columns='distance', inplace=True)
df.drop(columns='team', inplace=True)

#removes spaces and dashes from the dataset since they create errors
df.opp = df.opp.str.replace("'", "")
df.shot_type = df.shot_type.str.replace("-", "")
df.made = df.made.str.replace("True", "1")
df.made = df.made.str.replace("False", "0")

#creates a new column for each of the shot types and put a 1 under the column that corresponds to the shot used
df['3pointer'] = np.where(df.shot_type=='3pointer', 1,0 )
df['2pointer'] = np.where(df.shot_type=='2pointer', 1,0 )

#removes the secondaryType column since it is no longer needed to tell us the shot type
df.drop(columns='shot_type', inplace=True)

df = df.drop(df[df['player'] != "Anthony Davis"].index)

df['shotX'] = df['shotX'].astype(float)
df['shotY'] = df['shotY'].astype(float)

print(df)

def yreset(df):
    new_origins_old_x = 24

    return df.shotX - new_origins_old_x

def xreset(df):
    new_origins_old_y = 45.12

    return -(df.shotY - new_origins_old_y)

df['x'] = df.apply(xreset, axis=1)
df['y'] = df.apply(yreset, axis=1)

df.drop(columns='shotX', inplace=True)
df.drop(columns='shotY', inplace=True)


#finds distance of shots/goals based on the x and y of the shot
def dist(df):
    middle_goal_x = (90.24/2) - 4.56
    middle_goal_y = 0

    return math.sqrt((middle_goal_x - df.x)**2 + (middle_goal_y - df.y)**2)

#finds angle of shots/goals based on the x and y of the shot
def angle(df):
    middle_goal_x = (90.24/2) - 4.56
    middle_goal_y = 0
    adjacent = (middle_goal_y - df.y)

    if adjacent == 0:
        return 0
    else:
        return math.fabs(math.atan((middle_goal_x - df.x) / adjacent))
    
df['angle'] = df.apply(angle, axis=1)
df['dist'] = df.apply(dist, axis=1)

def buzzerbeat(df):
     
    secs = df.time_remaining.split(':')
    secs_remain = (float(secs[0])*60) + (float(secs[1]))

    clutch = df.score.split('-')
    clutched = int(clutch[0]) - int(clutch[1])
     
    if secs_remain <= 72 and df.quarter == "4th quarter" and clutched >= -3 and clutched <= 0:
        return 1
    else:
        return 0

def dstats(df):
    team_row = df_team[df_team['TEAM'] == df.opp].index.tolist()
    dstat = df_team.loc[team_row[0], 'DRTG']
    return dstat

df['dstats'] = df.apply(dstats, axis=1)
df['buzzerbeater'] = df.apply(buzzerbeat, axis=1)
df.drop(columns='time_remaining', inplace=True)
df.drop(columns='quarter', inplace=True)
df.drop(columns='score', inplace=True)
df.drop(columns='player', inplace=True)
df.drop(columns='opp', inplace=True)
df.drop(columns='x', inplace=True)
df.drop(columns='y', inplace=True)
df.dropna(inplace=True)
df['made'] = df['made'].astype(int)
df = df.reset_index()
del df['index']

df.to_csv("processed.csv")

print(df)
X = df.drop(columns='made')
y = df['made']

rus = RandomUnderSampler(sampling_strategy=1) # Numerical value
# rus = RandomUnderSampler(sampling_strategy="not minority") # String
X_rus, y_rus = rus.fit_resample(X, y)

#makes the number of goals and no goals in the data even to help give a more accurate model score
ros = RandomOverSampler(sampling_strategy=1) # String
X_ros, y_ros = ros.fit_resample(X, y)

#split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus.values, test_size=0.3, random_state=42)

# create model
lr_model = LinearRegression()
#fit model
lr_model.fit(X_train ,y_train)

predicted = lr_model.predict(X_test)

predicted = predicted.round().astype(int)

print("Score of the model is", r2_score(y_test, predicted))
print(lr_model.coef_) 
print(lr_model.intercept_)

#confusion matrix is much more accurate for showing accuracy
'''
TP = Predicted goal where there was  a goal
TN = Predicted no goal where there was no goal
FP = Predicted goal where there was no goal
FN = Predicted no goal where here was a goal

                 Actual Value
                    -------
                    |TP|FP|
    Predicted Value -------
                    |FN|TN|
                    -------
'''
cmatrix = confusion_matrix(y_test, predicted)
print(cmatrix)
#f1 score evaluates the accuracy by combinding the precision and recall(positives predicted as positives:total number of positives) of a model
f1 = f1_score(y_test, predicted)
print(f1)
#empty array that a prediction will be made off of based on the input data that will be added to this array
new_data = [[]]
new_cords_x = []
new_cords_y = []
running = 0
#loops the process of getting input and predicting
'''
I decided that it would be better to loop the whole input process
so that on the day of the exbition, I wouldn't have to run it each time.
'''
'''
while running == 0:
    time.sleep(5)
    new_data[0].clear()
    askshot = 0
    while askshot < 1:
        shot = input("Shot Type(2 pointer, 3 pointer):")
        if shot == "2 pointer":
            #list of data that tells the code that a wrist shot was used
            two_pointer = [0, 1]
            #adds the data to the empty array
            new_data[0].extend(two_pointer)
            #breaks the loop
            askshot += 1
        elif shot == "3 pointer":
            #list of data that tells the code that a snap shot was used
            three_pointer = [1, 0]
            #adds the data to the empty array
            new_data[0].extend(three_pointer)
            #breaks the loop
            askshot += 1
        #this last elif statement is if none of the inputs match any of the expected inputs
        elif shot != "2 pointer" or shot != "3 pointer":
            print("Please input one of the listed shot types.")
    
    #functinon that adds the x and y cords to the empty array from mouse click
    def mouse_event(event):
        new_st_x = event.xdata
        new_st_y = event.ydata
        
        #adds the x and y of each click to the lists new_cords_x and new_cords_y
        new_cords_x.append(new_st_x)
        new_cords_y.append(new_st_y)
        #loops the user until a proper shot type is inputed
       
    #displays an image of a hockey rink
    img = plt.imread('court.png')
    fig, ax = plt.subplots()
    plt.title("Click where you want the shot to be taken from. \nWhen done close this window.")
    #sets the size and cords of the court to match the size and cords used in the dataframe
    img = ax.imshow(img, extent=[-45.12, 45.12, -24, 24])
    #makes the hockey rink clickable and connects it to the mouse_event function that was made earlier
    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
    plt.axis("off")
    plt.plot()

    plt.show()

    #adds the last value of both lists, the most recent value, to the empty array
    new_data[0].append(new_cords_x[-1])
    new_data[0].append(new_cords_y[-1])
    
    middle_goal_x = (90.24/2) - 4.56
    middle_goal_y = 0

    adjacent =     adjacent = (middle_goal_y - new_data[0][3])

    if adjacent == 0:
        new_angle = 0
    else:
        new_angle = math.fabs(math.atan((middle_goal_x - new_data[0][2]) / adjacent))

    new_distance = math.sqrt((middle_goal_x - new_data[0][2])**2 + (middle_goal_y - new_data[0][3])**2)
    #adds the distance to the empty array
    new_data[0].append(new_distance)
    #adds the angle to the empty array
    new_data[0].append(new_angle)
    
    #variables used to break while loops
    askteam = 0
    #loops the user until a proper player name is inputed
    while askteam < 1:
        teamname = input("Defending Team(answer in abreviation form ie. GSW):")
        if teamname in set(df_team['TEAM']):
            #finds the row that the inputed players data is on and adds it to a list
            teamrow = df_team[df_team['TEAM'] == teamname].index.tolist()
            #locates the player_id of the inputed player from the row its on and the column its in
            new_dstat = df_team.loc[teamrow[0], 'DRTG']
            #breaks the loop
            askteam += 1
            #if the user input is not found in the fullName that means the player name was not spelled properly, or the player is not in the dataset
        elif teamname not in set(df_team['TEAM']):
            print("Please enter a different team.")
 
    #adds all of the data that was just found to the empty array
    new_data[0].append(new_dstat)

    #predicts if a goal would be scored based on the data added to the empty array
    new_pred = lr_model.predict(new_data)
    new_pred = new_pred.round().astype(int)
    new_prob = lr_model.predict(new_data)

    #if statement that prints goal if the model returns 1 or no goal if the model returns 0 along with the percentage of the goal going in
    if new_pred[0] == 1:
        print("goal with a ", new_prob[0][1], "%", "chance of going in")
    elif new_pred[0] == 0:
        print("no goal with a ", new_prob[0][1], "%", "chance of going in")
'''