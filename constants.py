from math import sqrt

class CONSTANT_VALUES:
  NAV2_DISCRETE_ACTIONS = { 0: [1,0], # east
                      1: [sqrt(2),sqrt(2)], # north east
                      2: [0,1], # north
                      3: [-sqrt(2),sqrt(2)], # north west
                      4: [-1,0], # west
                      5: [-sqrt(2),-sqrt(2)], # south west
                      6: [0,-1], # south
                      7: [sqrt(2),-sqrt(2)]} # south east