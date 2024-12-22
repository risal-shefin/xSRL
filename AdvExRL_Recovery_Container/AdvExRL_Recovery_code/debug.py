from math import *
discrete_action = { 0: [1,0], # east
                    1: [sqrt(2),sqrt(2)], # north east
                    2: [0,1], # north
                    3: [-sqrt(2),sqrt(2)], # north west
                    4: [-1,0], # west
                    5: [-sqrt(2),-sqrt(2)], # south west
                    6: [0,-1], # south
                    7: [sqrt(2),-sqrt(2)]} # south east
def eu_dist(a,b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
def binning_action(a):
    # action should be in terms of a = [ax, ay]
    # where ax, ay in range -1 to 1
    min_dist = [-1, 1]
    for k,v in discrete_action.items():
        if eu_dist(v, a) < min_dist[1]:
            min_dist[0] = k
            min_dist[1] = eu_dist(v,a)
    return min_dist[0]

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return sqrt(dotproduct(v, v))

def angle(v1, v2):
    angle = acos(dotproduct(v1, v2) / (length(v1) * length(v2))) / pi * 180
    if v2[1] < 0:
        angle = 360 - angle
    return angle

def round_state(s):
    rounded_s = []
    for ele in s:
        ele = round(ele , 2)
        rounded_s.append(ele)
    return rounded_s


tmp = set()
tmp.add((1,2))
print(tmp)
tmp.add((2,3))
print(tmp)
tmp.add((2,3))
print(tmp)
print(type((2,3)))

set2 = {1,2,3}
set2 = set2.union(tmp)
print(set2)
set2 = set2.union(tmp)
print(set2)
