# Decision making with Matrices
# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations. 
# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.  
# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.
 # This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems 
 # that are currently not leveraging machine learning.  
import numpy as np
#from sklearn.feature_extraction import DictVectorizer
from scipy.stats import rankdata
#%%
# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.  
people = {'Jane': {'willingness to travel': 5
                  ,'desire for new experience': 3
                  ,'cost': 8 
                  ,'indian food': 8 
                  ,'mexican food': 9
                  ,'hipster points': 1
                  ,'vegitarian':  5
                  }
          ,'Max': {'willingness to travel': 9
                  ,'desire for new experience': 6
                  ,'cost': 9 
                  ,'indian food': 3
                  ,'mexican food': 4
                  ,'hipster points': 1
                  ,'vegitarian':  1
                  }
		   ,'Anna Maria': {'willingness to travel': 9
                  ,'desire for new experience': 8
                  ,'cost': 4 
                  ,'indian food': 5
                  ,'mexican food': 6
                  ,'hipster points': 2
                  ,'vegitarian':  1
                  }
		     ,'Letizia': {'willingness to travel': 9
                  ,'desire for new experience': 8
                  ,'cost': 2 
                  ,'indian food': 8
                  ,'mexican food': 3
                  ,'hipster points': 5
                  ,'vegitarian':  9
                  }
			  ,'Daniele': {'willingness to travel': 6
                  ,'desire for new experience': 5
                  ,'cost': 7 
                  ,'indian food': 5
                  ,'mexican food': 8
                  ,'hipster points': 1
                  ,'vegitarian':  5
                  }
			  ,'Brooke': {'willingness to travel': 3
                  ,'desire for new experience': 3
                  ,'cost': 4 
                  ,'indian food': 9
                  ,'mexican food': 3
                  ,'hipster points': 7
                  ,'vegitarian':  8
                  }
			  ,'David': {'willingness to travel': 5
                  ,'desire for new experience': 3
                  ,'cost': 6
                  ,'indian food': 4
                  ,'mexican food': 8
                  ,'hipster points': 1
                  ,'vegitarian':  5
                  }
			  ,'Joe': {'willingness to travel': 9
                  ,'desire for new experience': 7
                  ,'cost': 1
                  ,'indian food': 8
                  ,'mexican food': 5
                  ,'hipster points': 1
                  ,'vegitarian':  5
                  }
			  ,'Diana': {'willingness to travel': 3
                  ,'desire for new experience': 2
                  ,'cost': 7
                  ,'indian food': 2
                  ,'mexican food': 5
                  ,'hipster points': 4
                  ,'vegitarian':  8
                  }
			  ,'Jeremy': {'willingness to travel': 5
                  ,'desire for new experience': 2
                  ,'cost': 2
                  ,'indian food': 6
                  ,'mexican food': 8
                  ,'hipster points': 1
                  ,'vegitarian':  2
                  }
          }          
#%%
# Transform the user data into a matrix(M_people). Keep track of column and row ids.   
M_people = np.zeros((len(people),7))
for i, p in enumerate(people):
	M_people[i,] = np.array(list(people[p].values()))

people_names = list(people)
people_cols =list(people[people_names[1]])

#%%


# Next you collected data from an internet website. You got the following information.

restaurants  = {'flacos':{'distance' : 3
                        ,'novelty' : 2
                        ,'cost': 1
                        ,'average rating': 5
                        ,'cuisine': 8
                        ,'vegitarians': 3
                        }
				  ,'Pizza Hut':{'distance' : 9
                        ,'novelty' : 1
                        ,'cost': 9
                        ,'average rating': 2
                        ,'cuisine': 2
                        ,'vegitarians': 4
                        }
				  ,'Flat Bread':{'distance' : 8
                        ,'novelty' : 4
                        ,'cost': 4
                        ,'average rating': 7
                        ,'cuisine': 8
                        ,'vegitarians': 6
                        }
				  ,'10 Barrels':{'distance' : 6
                        ,'novelty' : 6
                        ,'cost': 5
                        ,'average rating': 8
                        ,'cuisine': 7
                        ,'vegitarians': 4
                        }
				   ,'The Fork':{'distance' : 3
                        ,'novelty' : 9
                        ,'cost': 2
                        ,'average rating': 7
                        ,'cuisine': 8
                        ,'vegitarians': 8
                        }
}

#%%
# Transform the restaurant data into a matrix(M_resturants) use the same column index.

M_restaurants = np.zeros((len(restaurants),6))
for i, r in enumerate(restaurants):
	M_restaurants[i,] = np.array(list(restaurants[r].values()))

rest_names = list(restaurants)
rest_cols =list(restaurants[rest_names[1]])

#%%
# The most imporant idea in this project is the idea of a linear combination.  
# Informally describe what a linear combination is  and how it will relate to our resturant matrix.


#%%
# Choose a person and compute(using a linear combination) the top restaurant for them.  
# What does each entry in the resulting vector represent. 

np.dot(M_people[1,],M_restaurants)

#%%
# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent? 

# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?

# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   
# Do the same as above to generate the optimal resturant choice.  

# Why is there a difference between the two?  What problem arrives?  What does represent in the real world?

# How should you preprocess your data to remove this problem. 

# Find  user profiles that are problematic, explain why?

# Think of two metrics to compute the disatistifaction with the group.  

# Should you split in two groups today? 

# Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?

# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  
# Can you find their weight matrix? 



