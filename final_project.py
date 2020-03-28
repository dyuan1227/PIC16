"""
PIC 16 2018 Winter Final Project
Mengning(Maggie) Zhao, Rong Huang, Dongchen(Tony) Yuan
"""
import time
import nltk
from nltk import word_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community
'''
The community package implements community detection via modularity measure
Copyright (c) 2009, Thomas Aynaud <thomas.aynaud@lip6.fr>
All rights reserved.
'''


########################################################################################################################
#Reading all csv name files
begin = time.clock() #record start time

male_names= pd.read_csv('male_names.csv', ',')
female_names= pd.read_csv('female_names.csv', ',')
last_names= pd.read_csv('last_names.csv', ',')

male_name=male_names['Names'].values
female_name=female_names['Names'].values
last_name=last_names['Names'].values 
########################################################################################################################
#Book Testing, tokenization, creating a dictionary
# text_file =open("PP.txt", "r").read()
# paragraph=text_file.split('\r\r')             # Para is a list of strings(paragraphs)
# #print para
# lls=[[p] for p in paragraph]                  # List of lists of strings(paragraphs)
# tagged=[]
# for item in lls:
#     for i in item:
#         token=word_tokenize(i)           # tokenize inside each paragraph
#         tagged.append(nltk.pos_tag(token))
# print tagged                             #tagged is a list of list of strings with taggs

# #print tagged[0][0][1]
# my_dict={}
   
########################################################################################################################
#Book Testing, tokenization, creating a dictionary
text_file =open("PP.txt", "r").read()
paragraph=text_file.split('\r\r')             # Para is a list of strings(paragraphs)
#print para
tagged=[]
for item in paragraph:
    token=word_tokenize(item)           # tokenize inside each paragraph
    tagged.append(nltk.pos_tag(token))
print tagged                             #tagged is a list of list of strings with taggs

#print tagged[0][0][1]
my_dict={} 					   
########################################################################################################################
#Finding all matching names

for lst in tagged:
    for i in range(0,len(lst)):
        if lst[i][1]=="NNP":                                                # If the tagged is NNP
            key=lst[i][0]                                                   # We take this name as our key
            if ((key in male_name) or (key in female_name)):                # If this key is in our dictionary
                if (i<=len(lst)-2 and (lst[i+1][0] in last_name)):
                    key=key+" "+lst[i+1][0]
                if key in my_dict:                                          #We add the keys into out dictionary
                    my_dict[key] += 1
                else:
                    my_dict[key] = 1
print
print
print "Here is our dictionary with names and their counts"
print
print my_dict

# ########################################################################################################################                     
# #Find top ten keys
word = np.array(my_dict.keys())
count = np.array(my_dict.values())

for i in range(0,len(word)):
    if " " in word[i]:
        string=word[i]
        l=string.split()
        for item in l: #item 1: Harry item 2: Potter
            for j in range(0,len(word)): 
                if (i != j and item==word[j]):
                    count[i]=count[i]+count[j]
                    count[j]=0
                    word[j]=""

#print "hello"
n=0
while(n<len(count)):
    if count[n]==0:
        count = np.delete(count,n)
        word = np.delete(word,n)
    else:
        n=n+1

print
print
print "Here are list of names that match our findings:"
print
print word
print
print "At this point, we have added all counts of the first names and last names, which refers to the same person to that person"
print
print count




top = np.array([])
topcount = np.array([])

for i in range(10):
    max_index = np.argmax(count)
    top = np.append(top,word[max_index])
    topcount = np.append(topcount,count[max_index])
    
    word[max_index] = ''
    count[max_index] = 0
    

print
#print top
print
#print topcount

########################################################################################################################   
#initialize a adjacency matrix
adj = np.zeros((10,10))


for para in tagged: #for each paragraph
    name_index = set() #set list to identify all unique names in one paragraph
    for each in para: #for each word in the paragraph
        if (each[1]=="NNP"):
            for i in range(0,len(top)): #iterate the top list to find if NNP is a top name
                if (each[0] in top[i]):
                    name_index.add(i)  #if found, add index of top list
                    break
                
    name_index = list(name_index)
    #print name_index
                
    
    for i in range(0,len(name_index)):
        for j in range(i+1, len(name_index)):
            adj[name_index[i]][name_index[j]] +=1
            adj[name_index[j]][name_index[i]] +=1

#add the frequency counts to the adj matrix
for i in range(0,len(topcount)):
    #print topcount[i]
    adj[i][i] = topcount[i]

#print adj
########################################################################################################################   

G=nx.DiGraph(adj)

########################################################################################################################   
def pagerank(G, alpha=0.85,max_iter=100, tol=1.0e-6, weight='weight'):    
    if len(G) == 0:
        return {}
 
    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(G, weight=weight)
    N = W.number_of_nodes()
 
    # Choose fixed starting vector
    x = dict.fromkeys(W, 1.0 / N)

    # Assign uniform personalization vector if not given
    p = dict.fromkeys(W, 1.0 / N)

    dangling_weights = p
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]
 
    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]
 
        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N*tol:
            return x
#    raise NetworkXError('pagerank: power iteration failed to converge '
#                        'in %d iterations.' % max_iter)

p_rank=pagerank(G).values()
for i in range(0,len(p_rank)):
    p_rank[i]*=3000

print "Page Rank algorithm provides us the values of the top ten chracters:"
print
print p_rank
print
######################################################################################################################## 

#community detection_ modularity
H=G.to_undirected()
communities = community.best_partition(H)
global_modularity = community.modularity(communities, H)
print(global_modularity)
values = [communities.get(node) for node in H.nodes()]


#edges
all_weights = []
for (node1,node2,data) in G.edges(data=True):
    all_weights.append(data['weight']) #we'll use this when determining edge thickness

print all_weights
    #Plot the edges - one by one
    
#p_rank_std=p_rank/p_rank[0]
  

    
pos=nx.spring_layout(G) 
labeldict = {}                      #dictionary of node to node names

for i in range(0,len(top)):
    labeldict[i] = top[i]
for weight in all_weights:
    #Form a filtered list with just the weight you want to draw
    weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
    #multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
    width = weight/45
    
    nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width,edges_color=values)
nx.draw_networkx_nodes(G,pos,node_color=values,node_size=p_rank,with_labels = True)

# customize positions of labels and font size
pos_new = {}
for k, v in pos.items():
    pos_new[k] = (v[0], v[1]-0.13)
    
nx.draw_networkx_labels(G,pos=pos_new,labels=labeldict,
                        font_size=14, 
                        font_family='Arial')


#change labels 
plt.axis('off')

plt.show() 
########################################################################################################################   



#print len(p_rank)


end = time.clock()
print end - begin #calculate difference (elapsed time)
#Page Rank


