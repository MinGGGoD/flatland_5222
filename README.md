# Welcome to piglet

## Requirement

python >= 3.6

## Install

1. Clone the repo to your machine
2. Run:

```
$ python setup.py install
```

## Usage

### Commandline Interface

```
$ piglet.py --help
```

run a scenario:
```
$ python3 piglet.py -p ./example/example_n_puzzle_scenario.scen -f graph -s uniform  
```

### Piglet Library
piglet provides a variety of flexible search algorithms. These algorithms are 
able to help you to build your application.

#### Example 

To use an algorithm you need a domain instance, an expander instance and a search instance. 
```python
import os,sys
from lib_piglet.domains import gridmap
from lib_piglet.expanders.grid_expander import grid_expander
from lib_piglet.search.tree_search import tree_search
from lib_piglet.utils.data_structure import bin_heap,stack,queue

mapfile = "./example/gridmap/empty-16-16.map"

# create an instance of gridmap domain
gm = gridmap.gridmap(mapfile)

# create an instance of grid_expander and pass the girdmap instance to the expander.
expander = grid_expander(gm)

# create an instance of tree_search, and pass an open list (we use a binary heap here)
# and the expander to it.
search = tree_search(bin_heap(), expander)

# start search by proving a start state and goal state. For gridmap a state is a (x,y) tuple 
solution = search.get_path((1,2),(10,2))

# print solution
print(solution)

```


```
level0: 10*10, 5 agents;     40
level1: 25*25, 12 agents;    96
level2: 50*50, 25 agents;    200
level3: 75*75, 37 agents;    296
level4: 100*100, 50 agents;  400
level5: 150*150, 75 agents;  600
level6: 150*150, 150 agents; 1200

total agents: 40 + 96 + 200 + 296 + 400 + 600 + 1200 = 2832
```

Test case      | Total agents | Agents done  | DDLs met     | Plan Time  | SIC          | Makespan     | Penalty      | Final SIC    | P Score     
level0_test_0  | 5            | 5            | 4            | 0.0        | 68           | 25           | 0            | 68           | 13          
level0_test_1  | 5            | 5            | 5            | 0.01       | 69           | 27           | 0            | 69           | 13          
level0_test_2  | 5            | 5            | 4            | 0.01       | 76           | 28           | 5            | 81           | 16          
level0_test_3  | 5            | 5            | 5            | 0.01       | 87           | 32           | 0            | 87           | 17          
level0_test_4  | 5            | 5            | 4            | 0.01       | 68           | 21           | 0            | 68           | 13          
level0_test_5  | 5            | 5            | 5            | 0.01       | 65           | 19           | 0            | 65           | 13          
level0_test_6  | 5            | 5            | 5            | 0.01       | 58           | 22           | 0            | 58           | 11          
level0_test_7  | 5            | 5            | 4            | 0.01       | 36           | 12           | 1            | 37           | 7           
level1_test_0  | 12           | 12           | 11           | 0.06       | 522          | 82           | 1            | 523          | 43          
level1_test_1  | 12           | 12           | 8            | 0.11       | 636          | 115          | 61           | 697          | 58          
level1_test_2  | 12           | 12           | 12           | 0.05       | 385          | 50           | 0            | 385          | 32          
level1_test_3  | 12           | 12           | 11           | 0.06       | 424          | 87           | 0            | 424          | 35          
level1_test_4  | 12           | 12           | 10           | 0.13       | 578          | 85           | 25           | 603          | 50          
level1_test_5  | 12           | 10           | 7            | 0.16998    | 1227         | 399          | 656          | 1883         | 156         
level1_test_6  | 12           | 12           | 7            | 0.13       | 660          | 86           | 44           | 704          | 58          
level1_test_7  | 12           | 12           | 9            | 0.18       | 755          | 162          | 115          | 870          | 72          
level2_test_0  | 25           | 25           | 21           | 0.35       | 1692         | 127          | 47           | 1739         | 69          
level2_test_1  | 25           | 24           | 22           | 0.5        | 2702         | 799          | 558          | 3260         | 130         
level2_test_2  | 25           | 24           | 19           | 0.53       | 2827         | 799          | 656          | 3483         | 139         
level2_test_3  | 25           | 25           | 21           | 0.54       | 1768         | 142          | 32           | 1800         | 72          
level2_test_4  | 25           | 25           | 19           | 0.96       | 2571         | 171          | 258          | 2829         | 113         
level2_test_5  | 25           | 21           | 17           | 0.860002   | 4683         | 799          | 2718         | 7401         | 296         
level2_test_6  | 25           | 21           | 20           | 1.290005   | 4670         | 799          | 2753         | 7423         | 296         
level2_test_7  | 25           | 25           | 21           | 0.86       | 2279         | 220          | 158          | 2437         | 97          
level3_test_0  | 37           | 34           | 33           | 1.92       | 7984         | 1199         | 3005         | 10989        | 297         
level3_test_1  | 37           | 36           | 33           | 1.650001   | 5214         | 1199         | 983          | 6197         | 167         
level3_test_2  | 37           | 37           | 32           | 1.27       | 3802         | 177          | 101          | 3903         | 105         
level3_test_3  | 37           | 37           | 33           | 2.24       | 4252         | 249          | 58           | 4310         | 116         
level3_test_4  | 37           | 30           | 27           | 3.37001    | 11398        | 1199         | 7225         | 18623        | 503         
level3_test_5  | 37           | 37           | 28           | 2.32       | 4842         | 342          | 216          | 5058         | 136         
level3_test_6  | 37           | 37           | 29           | 4.01       | 4877         | 251          | 201          | 5078         | 137         
level3_test_7  | 37           | 33           | 28           | 4.88001    | 9715         | 1199         | 4286         | 14001        | 378         
level4_test_0  | 50           | 50           | 45           | 4.92       | 7603         | 314          | 49           | 7652         | 153         
level4_test_1  | 50           | 45           | 42           | 6.82       | 14286        | 1599         | 7061         | 21347        | 426         
level4_test_2  | 50           | 50           | 48           | 5.35       | 7564         | 230          | 19           | 7583         | 151         
level4_test_3  | 50           | 48           | 46           | 4.78       | 11260        | 1599         | 2906         | 14166        | 283         
level4_test_4  | 50           | 33           | 30           | 103.7613   | 31731        | 1599         | 23882        | 55613        | 1112        
level4_test_5  | 50           | 45           | 39           | 74.31      | 14869        | 1599         | 7115         | 21984        | 439         
level4_test_6  | 50           | 50           | 43           | 8.1        | 7083         | 308          | 87           | 7170         | 143         
level4_test_7  | 50           | 50           | 45           | 3.79       | 6223         | 214          | 45           | 6268         | 125         
level5_test_0  | 75           | 72           | 63           | 15.29      | 21280        | 2399         | 6540         | 27820        | 370         
level5_test_1  | 75           | 72           | 70           | 11.25      | 20709        | 2399         | 6511         | 27220        | 362         
level5_test_2  | 75           | 75           | 72           | 20.13      | 15460        | 452          | 45           | 15505        | 206         
level5_test_3  | 75           | 70           | 67           | 11.05      | 24668        | 2399         | 10495        | 35163        | 468         
level5_test_4  | 75           | 60           | 54           | 42.41      | 48082        | 2399         | 30704        | 78786        | 1050        
level5_test_5  | 75           | 73           | 65           | 19.97      | 18671        | 2399         | 4609         | 23280        | 310         
level5_test_6  | 75           | 66           | 61           | 28.58      | 33910        | 2399         | 19117        | 53027        | 707         
level5_test_7  | 75           | 65           | 61           | 31.23      | 37399        | 2399         | 22671        | 60070        | 800         
level6_test_0  | 150          | 149          | 143          | 26.85      | 27375        | 2399         | 2195         | 29570        | 197         
level6_test_1  | 150          | 145          | 137          | 23.37      | 35690        | 2399         | 11176        | 46866        | 312         
level6_test_2  | 150          | 150          | 143          | 37.63      | 26521        | 524          | 130          | 26651        | 177         
level6_test_3  | 150          | 141          | 136          | 243.38     | 45458        | 2399         | 19217        | 64675        | 431         
level6_test_4  | 150          | 144          | 131          | 76.3       | 42007        | 2399         | 13318        | 55325        | 368         
level6_test_5  | 150          | 144          | 136          | 43.03      | 41200        | 2399         | 17658        | 58858        | 392         
level6_test_6  | 150          | 133          | 123          | 59.38      | 64442        | 2399         | 38288        | 102730       | 684         
level6_test_7  | 150          | 145          | 140          | 47.45      | 35234        | 2399         | 11083        | 46317        | 308         
Summary        | 2832 (sum)   | 2680 (sum)   | 2454(sum)    | 977.68(sum) | 719715 (sum) | 52947 (sum)  | 279084 (sum) | 998799 (sum) | None (final)