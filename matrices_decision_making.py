# Decision making with Matrices
# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations. 
# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.  
# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.
 # This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems 
 # that are currently not leveraging machine learning.  
import numpy as np
import math
#from sklearn.feature_extraction import DictVectorizer
import scipy.stats as ss
#%% Functions
def getRank(data):
	return(ss.rankdata(len(data)-ss.rankdata(data)+2,method='min'))

#%%
# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.  
 
people = {'Jane': {'willingness to travel': 5
                  ,'desire for new experience': 2
                  ,'cost': 5 
                  #,'indian food': 8 
                  #,'mexican food': 9
                  #,'hipster points': 1
                  ,'vegitarian':  2
                  }
          ,'Max': {'willingness to travel':5
                  ,'desire for new experience': 2
                  ,'cost': 5 
                  #,'indian food': 3
                  #,'mexican food': 4
                  #,'hipster points': 1
                  ,'vegitarian':  1
                  }
		   ,'Anna Maria': {'willingness to travel': 4
                  ,'desire for new experience': 3
                  ,'cost': 3 
                  #,'indian food': 5
                  #,'mexican food': 6
                  #,'hipster points': 2
                  ,'vegitarian':  1
                  }
		     ,'Letizia': {'willingness to travel': 4
                  ,'desire for new experience': 3
                  ,'cost': 1 
                  #,'indian food': 8
                  #,'mexican food': 3
                  #,'hipster points': 5
                  ,'vegitarian':  3
                  }
			  ,'Daniele': {'willingness to travel': 3
                  ,'desire for new experience': 2
                  ,'cost': 3
                  #,'indian food': 5
                  #,'mexican food': 8
                  #,'hipster points': 1
                  ,'vegitarian':  2
                  }
			  ,'Brooke': {'willingness to travel': 1
                  ,'desire for new experience': 2
                  ,'cost': 3 
                  #,'indian food': 9
                  #,'mexican food': 3
                  #,'hipster points': 7
                  ,'vegitarian':  5
                  }
			  ,'David': {'willingness to travel': 3
                  ,'desire for new experience': 2
                  ,'cost': 3
                  #,'indian food': 4
                  #,'mexican food': 8
                  #,'hipster points': 1
                  ,'vegitarian':  2
                  }
			  ,'Joe': {'willingness to travel': 5
                  ,'desire for new experience': 4
                  ,'cost': 1
                  #,'indian food': 8
                  #,'mexican food': 5
                  #,'hipster points': 1
                  ,'vegitarian':  4
                  }
			  ,'Diana': {'willingness to travel': 2
                  ,'desire for new experience': 2
                  ,'cost': 4
                  #,'indian food': 2
                  #,'mexican food': 5
                  #,'hipster points': 4
                  ,'vegitarian':  5
                  }
			  ,'Jeremy': {'willingness to travel': 3
                  ,'desire for new experience': 1
                  ,'cost': 1
                  #,'indian food': 6
                  #,'mexican food': 8
                  #,'hipster points': 1
                  ,'vegitarian':  1
                  }
          }   


              
#%%
# Transform the user data into a matrix(M_people). Keep track of column and row ids.   

people_names = list(people)
people_cols =list(people[people_names[1]])

M_people = np.zeros((len(people_names),len(people_cols)))
for i, p in enumerate(people):
	M_people[i,] = np.array(list(people[p].values()))

print(M_people)
#%%

# Next you collected data from an internet website. You got the following information.
            
restaurants  = {'flacos':{'distance' : 2
                        ,'novelty' : 1
                        ,'cost': 1
                        #,'average rating': 5
                        #,'cuisine': 8
                        ,'vegitarians': 5
                        }
				  ,'Pizza Hut':{'distance' : 4
                        ,'novelty' : 1
                        ,'cost': 5
                        #,'average rating': 2
                       # ,'cuisine': 2
                        ,'vegitarians': 2
                        }
				  ,'Flat Bread':{'distance' : 4
                        ,'novelty' : 2
                        ,'cost': 2
                        #,'average rating': 7
                        #,'cuisine': 8
                        ,'vegitarians': 3
                        }
				  ,'10 Barrels':{'distance' : 3
                        ,'novelty' : 2
                        ,'cost': 3
                        #,'average rating': 8
                        #,'cuisine': 7
                        ,'vegitarians': 2
                        }
				   ,'The Fork':{'distance' : 2
                        ,'novelty' : 5
                        ,'cost': 1
                        #,'average rating': 7
                        #,'cuisine': 8
                        ,'vegitarians': 4
                        }
                   ,"Max's House":{'distance' : 5
                        ,'novelty' : 1
                        ,'cost': 5
                        #,'average rating': 9
                        #,'cuisine': 9
                        ,'vegitarians': 2
                        }
                   ,'Costco':{'distance' : 3
                        ,'novelty' : 1
                        ,'cost': 4
                        #,'average rating': 9
                        #,'cuisine': 9
                        ,'vegitarians': 1
                        }
                   ,'Bonefish':{'distance' : 3
                        ,'novelty' : 5
                        ,'cost': 2
                        #,'average rating': 9
                        #,'cuisine': 9
                        ,'vegitarians': 5
                        }
}

#%%
# Transform the restaurant data into a matrix(M_resturants) use the same column index.

rest_names = list(restaurants)
rest_cols =list(restaurants[rest_names[1]])

M_restaurants = np.zeros((len(rest_names),len(rest_cols)))
for i, r in enumerate(restaurants):
	M_restaurants[i,] = np.array(list(restaurants[r].values()))

print(M_restaurants)
#%% QUESTION 1
# The most imporant idea in this project is the idea of a linear combination.  
# Informally describe what a linear combination is  and how it will relate to our resturant matrix.
"""
We linear combination  we create an expression that combine multipls terms (x) and weights (betas) to obtain a final result (y). 
The final result has memory of each betas per each x, henve is similar to a weighted max, sum, or average of such terms.
In this case we are can use the linear combination to evalute the restaurant that has the highest value based on the resturant rating (x) 
peple preferences (betas).

All Calculations are verified using the attahed HW3_Verify.xlsx Excel file
"""
#%% QUESTION 2
# Choose a person and compute(using a linear combination) the top restaurant for them.  
# What does each entry in the resulting vector represent. 

# Choosing Max
p = people_names.index('Max')
out=np.dot(M_people[p,],M_restaurants.T)


print(out)
"""
Each entry in the output represent the total weightes score of each restaurant based on the restarant rating and Max's preferences. 
The highest the value the most likely the restauran is a good choice for Max
"""

outRank = getRank(out)

print("Best Restaurant for", people_names[p],"is",rest_names[np.flatnonzero(outRank ==1)[0]]," witha score of ",out[np.flatnonzero(outRank ==1)[0]])

#%% QUESTION 3
# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent? 

#M_usr_x_rest=np.dot(M_people,M_restaurants.T)
M_usr_x_rest=np.dot(M_restaurants,M_people.T)
#np.linalg.lstsq(M_restaurants,M_usr_x_rest.T)[0].T
print(M_usr_x_rest)
print("Rows is restaurants: ",print(rest_names)) ##
print("Column is people: ",print(people_names)) ##

"""
The a_ij matrix represent the value of each resturant's scores weighted by the preferences of each person.
For our problem it represent how much a restaurant (j) is good for the person (i).The highest the value the better the restaurant j is for the person i
Example the position (1,1) is the value of restauran flacos for Jane
"""

#%% QUESTION 4
# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
rest_score  = M_usr_x_rest.sum(axis=1) 
c=0
for i in reversed(np.argsort(rest_score)):
    c=c+1
    print(c,". Restaurant",rest_names[i], " tot Score:", rest_score[i])  

rest_rank = getRank(rest_score)
rest_best_id = np.flatnonzero(rest_rank ==1)[0]
rest_best_score = rest_score[rest_best_id]
rest_best_name = rest_names[rest_best_id]
 
# Each entry represent the total value per each restaurant across all people 
print("Best Restaurant for all people (based on Scores) is",rest_best_name,"with a score of", rest_best_score)

#%% QUESTION 5
# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   
# Do the same as above to generate the optimal resturant choice.  

M_usr_x_rest_rank = np.zeros_like(M_usr_x_rest)

for r in range(M_usr_x_rest.shape[1]):
    M_usr_x_rest_rank[:,r] = getRank(M_usr_x_rest[:,r])
#M_usr_x_rest_rank = [getRank(M_usr_x_rest[x,:]) for x in range(M_usr_x_rest.shape[0])]

rest_score2   = np.sum(M_usr_x_rest_rank,axis=1)
c=0
for i in (np.argsort(rest_score2)):
    c=c+1
    print(c,". Restaurant",rest_names[i], " tot Score:", rest_score2[i]) #,"(", (120- rest_score2[i]),")")  

rest_rank2 = ss.rankdata(rest_score2,method='min')
rest_best_id2 = np.flatnonzero(rest_rank2 ==1)[0]
rest_best_score2 = rest_score2[rest_best_id2]
rest_best_name2 = rest_names[rest_best_id2]

print("Best Restaurant for all people (based on Ranks) is",rest_best_name2,"with a score of", rest_best_score2)


#%% QUESTION 6
# Why is there a difference between the two?  What problem arrives?  What does represent in the real world?
"""
The first method is summing all the scores each restaurant received from people, this 
   maintains the range and skewness of the scores.   (Example: Rest1 received a score of 100 and 1000 = tot score 1100,
   while rest2 got a score of 200 and 150 = tot score 350)

The second method is summing the ranks each restaurant received from the people. This removes the skewness and range 
   of the scores. (Example: rest 1 received ranks of 2 and  1 = tot score = 3, rest 2 received ranks of 1  and 2 = tot score 3)

The problem is that the first model is sensitive to outliers on the scores (scores are not ordinal), 
   while the second method is using ordinal vlaues (rank) and is ignoring the magnitude of the socres.

So, in the real world, a person can influence the ranking of the first model by creating a wide difference in the scores across restaurants. (rest1=100, rest2=1000)

The ranked based model (model 2) removes such problem, but is ignoring the magnitude of the scoring. If two restaurants 
   have a very similar score, may receive a very distand rankings (while we want them close on the ranking)

"""
#%% QUESTION 7
# How should you preprocess your data to remove this problem. 
"""
The problem  is caused by the skewness of the scores made by each person. 
A good method to remove skewness is to LOG-transform the score. The Log will make the scoring distribution more normal, hence
  the final ranking can still be basesd on scores (method 1) and we have mitigate extreme skewess (or bias) of the data.
  
"""

rest_score_log  = [np.log(M_usr_x_rest[x,:]).sum() for x in range(M_usr_x_rest.shape[0])]
c=0
for i in reversed(np.argsort(rest_score_log)):
    c=c+1
    print(c,". Restaurant",rest_names[i], " tot Score:", round(rest_score_log[i],3))  

rest_rank_log = getRank(rest_score_log)
rest_best_id_log = np.flatnonzero(rest_rank_log ==1)[0]
rest_best_score_log = rest_score_log[rest_best_id_log]
rest_best_name_log = rest_names[rest_best_id_log]
 
# Each entry represent the total value per each restaurant across all people 
print("Best Restaurant for all people (based on Logged Scores) is",rest_best_name_log,"with a score of", round(rest_best_score_log,3))

#%% QUESTION 8
#  Find  user profiles that are problematic, explain why?

"""
The final ranking with the first model shows Max's House as the best restaurant while the second method shows Bonefish as best restaurant.

Looking at the data we see that Bonefish receiced very high scores, but was not always the first choise. Max's House was first most of 
   the times, evenf if its scores were not very high.

"""
_=rest_names.index(rest_best_name)
print(rest_best_name ," scores",M_usr_x_rest[_,:]," Final Rank Method 1-Scores: ",rest_rank[_])
print(rest_best_name, " Ranks",M_usr_x_rest_rank[_,:]," Final Rank Method 2-Ranks: ",rest_rank2[_])
print('-'*10)
_=rest_names.index(rest_best_name2)
print(rest_best_name2, " scores",M_usr_x_rest[_,:]," Final Rank Method 1-Scores: ",rest_rank[_])
print(rest_best_name2, " ranks",M_usr_x_rest_rank[_,:]," Final Rank Method 2-Ranks: ",rest_rank2[_])


#%% QUESTION 9

#  Think of two metrics to compute the disatistifaction with the group.  

# 1) People Disatisfaction  vs. the Selected Restaurant

"""
1) We caculate the People Disatisfaction vs. the Selected Restaurant.
	We can calculate the difference between each person score for the best score received by selected restaurant.
	A measure of 0 means the person has the max satisfaction for the selected restaurant, any number 
    higher measure disattisfaction.
	
"""
rest_best_max_score = M_usr_x_rest[rest_best_id,:].max()
#rest_best_people_score  =  [M_usr_x_rest[rest_best_id,].sum() for x in range(M_usr_x_rest.shape[1])]
people_dissatisfaction =  abs(M_usr_x_rest[rest_best_id,:]   - rest_best_max_score)
print("List of people from the most disattisfied to the least disattisfied about the choise of",rest_best_name,"as best restaurant")
i=0
for x in reversed(np.argsort(people_dissatisfaction) ) :
    i=i+1
    print(i,". ",people_names[x],"disattisfaction:",people_dissatisfaction[x])
#------------
print("-"*20)
rest2_best_max_score = M_usr_x_rest[rest_best_id2,:].max()
people_dissatisfaction2 =  abs(M_usr_x_rest[rest_best_id2,:]   - rest2_best_max_score)
print("List of people from the most disattisfied to the least disattisfied about the choise of",rest_best_name2,"as best restaurant")
i=0
for x in reversed(np.argsort(people_dissatisfaction2) ) :
    i=i+1
    print(i,". ",people_names[x],"disattisfaction:",people_dissatisfaction2[x])
    
print("We can measure the overall disattisfaction in the group by using a standard deviation of above values")
print(np.std(people_dissatisfaction2))
print("we have a wide SD, hence we can conclude there is a strong disatisfaction within the group on the selected restaurant")

#%% 2) Overall Disatisfaction for all Restaurant vs. People preferences
"""
2) We can calculate the Disatisfaction score each  restaurant has. This metrics shows how much each person is satisfied or not with the decision
       to choose or not a specific restaurant. 
       Restaurants with low disattisfaction score and low overall score should be excluded from the list end eventually replaced with
       other options, as no one is happy with them. Restaurants with low disattisfaction score and high overall score are the ideal candidates 
       for best place to go.
       Restaurants with high disattisfaction score are the risky. Choosing to visit or to remove a restaurant with high disatisfaction can
       bring  division in the group as there is not an good agreement on their final score.

      We can calcualte the disattisfaction score by calculating the average of the difference between  each restaurant mean score (across all 
      people) vs. each  person score for that restaurant. 
   
    
"""

rest_mean_score = M_usr_x_rest.mean(axis=1)

rest_people_disattisfaction  =  np.mean(abs(M_usr_x_rest.T - rest_mean_score),axis=0)
rest_people_disattisfaction_sd  =  np.std(abs(M_usr_x_rest.T - rest_mean_score),axis=0)

print("List of restaurants from the most disattisfaction to the least disattisfaction vs. people preference")
i=0
for x in reversed(np.argsort(rest_people_disattisfaction) ) :
    i=i+1
    print(i,".",rest_names[x],"overall disattisfaction:",round(rest_people_disattisfaction[x],2)," (Std:",round(rest_people_disattisfaction_sd[x],3),").")


print("We can see Bonefish and Max's House habe the highest disatisfaction score even if it has the highest overall score. This indicates\
      that there is an overall disattisfaction within the team. Splitting the people in groups, and choose a best restaurant \
      per each group may help to reduce the ovreal disatisfaction")

#%% QUESTION 10
# Should you split in two groups today? 
"""
As hinted by the previous questions, we may have groups of people with similar taste but different from other groups. 
This would lead to some level of disattisfction across the team if we choose only one best restaurant.
We may try to see if there is are two group of people that share similar rating but very differnt between them. If so,
we can and create a spearate ranking of restaurants for them. 

If the grouping is effective we should reduce the disattisfaction score

"""
from sklearn.cluster import KMeans
X = M_usr_x_rest.T
k=2
km = KMeans(n_clusters=k, random_state=0,n_init=10,max_iter=300).fit(X)

def bestRest(data):
    rest_score_  = np.sum((data),axis=1)
    print(rest_score_)
       
    rest_rank_  = getRank(rest_score_)
    rest_best_id_  = np.flatnonzero(rest_rank_ ==1)[0]
    rest_best_score_  = rest_score_log[rest_best_id_]
    rest_best_name_  = rest_names[rest_best_id_]
     
    # Each entry represent the total value per each restaurant across all people 
    rest_mean_score_ = (data[rest_best_id_,:]).mean()
    
    diss  =  np.mean(abs(data[rest_best_id_].T - rest_mean_score_),axis=0)
    diss_sd  =  np.std(abs(data[rest_best_id_].T - rest_mean_score_),axis=0)
    #diss =  abs(data.mean(axis=1) - data.max(axis=1))[rest_best_id_]
    
    print("Best Restaurant for this group is ",rest_best_name_,"with a score of", round(rest_best_score_,3)
    ," disattisfaction score:",round(diss,2)," (std:",round(diss_sd,3),")")
    return(rest_best_id_)
    
print("-"*20)

for i in range(k):
    print("CLUSTER ",i)
    data=np.dot(M_restaurants,M_people[km.labels_ == i].T)   
    
    print("people:", data.shape[1]," ->",np.where(km.labels_==i))
    _=bestRest(data)
    print("      Original Disattisfaction Score for ",rest_names[_],":",rest_people_disattisfaction[_]," (std:",rest_people_disattisfaction_sd[_],").")
    print("-"*20)

print("We can see we have two different restaurants selected for the two groups. In both situation the disattisfaction score for the \
      selected restaurant has been reduced. The stardard deviations also reduced.")
 

 

#%% QUESTION 11
# Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?
"""
It depends if we have a constrained budget or not. If we have higher budget, the weight associate with cost preference (people matrix) 
may be lowerd. Otherwise, if we have a constrained budget (like the boss is paying from his/her money) we need to increase the
cost weight in the poeple matrix.

For this exercise we are assuming the boss is paying from her money, hence we want to be cost conscious. We proceed by
adjusting the cost weight on the poeple matrix to 5
"""
#%%
#recalculate the best resdt
M_people2 = M_people.copy()
M_people2[:,people_cols.index('cost')]=5
M_usr_x_rest2=np.dot(M_people2,M_restaurants.T)

rest2_score  = [M_usr_x_rest2[:,x].sum() for x in range(M_usr_x_rest2.shape[1])]

for i in np.argsort(rest2_score):
    print("Restaurant",rest_names[i], " tot Score:", rest2_score[i])  

rest2_rank = getRank(rest2_score)
rest2_best_id = np.flatnonzero(rest2_rank ==1)[0]
rest2_best_score = rest2_score[rest2_best_id]
rest2_best_name = rest_names[rest2_best_id]

print("Best Restaurant for all people is",rest2_best_name,"with a score of", rest2_best_score)
"""
Calculation is verified using the HW3_Verify.xlsx Excel file
"""

#%% QUESTION 12

# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  
# Can you find their weight matrix? 

"""
We cannot recreate the weight matrix with just the rank, but we can do with the final decision matrix (usr_x_rest)
Calculation is verified using the HW3_Verify.xlsx Excel file
"""
#Assuming we have the following decision matrix for the restarants
new_team_x_rest =  M_usr_x_rest

new_team_matrix= np.linalg.lstsq(M_restaurants,new_team_x_rest,rcond=-1)[0].T

print("New Team Weight Matrix is:")
print(new_team_matrix)
for i,p in enumerate(people_names): print(p,"weights:",new_team_matrix[i,:])
print("Columns Name:")
print(people_cols)
#%%
"""
If we have the final score per each restaurant (the x_rest vector) we can calculate the total of the peple weights 
per each crieria (x_people)
Calculation is verified using the HW3_Verify.xlsx Excel file
"""
new_team_rest_score = np.sum(new_team_x_rest,axis=1)
print("Final Score per each restarant")
for i,s in enumerate(new_team_rest_score): print(rest_names[i],"tot weight:",round(s,3))
new_team_tot_weights = np.linalg.lstsq(M_restaurants,new_team_rest_score.T,rcond=-1)[0]
print("\nSum of people weights per each criteria")
for i,s in enumerate(new_team_tot_weights): print(people_cols[i],"tot weight:",round(s,3))

