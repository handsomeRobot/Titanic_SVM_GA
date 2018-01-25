
#coding=utf-8



#Import necessary modules
import random



#Based on the pre-generation and their performances, generate a new
#generation by 


#Given finess list, return the selected index
def roulette_wheel(fitness_ls):
    #If the fitness_ls = [1, 3, 2], the roulette_wheel is like [0, 1, 4, 6]
    roulette_wheel = [0]
    for i, fitness in enumerate(fitness_ls):
        roulette_wheel.append(fitness + roulette_wheel[i])
    #Rotate the wheel
    pointer = random.randint(0, int(roulette_wheel[-1]))
    for i, num in enumerate(roulette_wheel):
        if roulette_wheel[i] <= pointer and pointer <= roulette_wheel[i + 1]:
            return(i)
        
        
        

def feature_selection(population, num_features, feature_combines, fitness_ls):
    
    #Initialize the population if is the first generation
    if feature_combines == [] and fitness_ls == []:
        for i in range(population):
            #Lowest possible for feature_combine, i.e., 0
            low = 0
            #Highest possible for feature_combine, i.e., 3 if there are 
            #two feature, the corresponding binary form is 11
            high = 2 ** num_features - 1
            feature_combine = bin(random.randint(low, high))[2:] #i.e., '0b11' to '11'
            feature_combine = '0' * (num_features - len(feature_combine)) + feature_combine
            map_combine = [i == '1' for i in feature_combine]
            feature_combines.append(map_combine)
        return(feature_combines)
    
    #Normalize the fitness_ls:
    fitness_ls = [((i / sum(fitness_ls)) * 100) ** 2 for i in fitness_ls]
    
    
    #Roulette wheel selection based on fitness_ls
    #Here we select N/2 population, into N/4 pairs, each pair generates 4 offsprings
    #So the next population remains the same N
    selections = []
    for i in range(int(population / 2)):
        try:
            index = roulette_wheel(fitness_ls)
            selections.append(feature_combines[index])
        except Exception as e:
            print(e)
            print('The chosen index by roulette_wheel is: ', index)
    
    #Crossover to generate offsprings
    #The pairing and crossover is a random-wise
    offsprings = []
    while len(selections) > 1:
        random.shuffle(selections)
        father = selections.pop()
        mother = selections.pop()
        for g in range(4):
            offspring = [0] * len(father)
            for i, bit in enumerate(offspring):
                offspring[i] = [father, mother][random.randint(0, 1)][i]
            offsprings.append(offspring)
            
    #Mutation one bit, 50% probality
    #avoid 0000, which means there is no feature selected
    for offspring in offsprings:
        index = random.randint(0, len(offspring) - 1)
        offspring[index] = random.randint(0, 1)
        while 1 not in offspring:
            index = random.randint(0, len(offspring) - 1)
            offspring[index] = random.randint(0, 1)
        
        
    #Return the result
    #[0110] = [False, True, True, False]
    for i, offspring in enumerate(offsprings):
        offsprings[i] = [i == 1 for i in offspring]
    return(offsprings)

