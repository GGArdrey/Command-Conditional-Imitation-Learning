'''
This file contains the routes for the CARLA Benchmarks. Those are spawn point IDs on different maps
NOTE: The routes provided here will not lead to reproducable results. Spawn Points where manually shifted around UE4 Editor
to allow smooth routes. Further, spawn points are not receiving the same IDs over different CARLA Server startups,
hence they are sorted by x/y coordinates in the CarlaEnvironment Class

If you intend to re-create benchmark routes from the Thesis, please manually shift the spawn points in the UE4 Editor
create your own lists
'''


# town 7 opt
def town7_opt():
    # outside route, counter clockwise
    return [6,2,0,1,8,47,109,108,70,15]


def town7_opt3():
    return [45, 3, 89, 24, 76, 18, 91, 33, 68, 105, 7, 26, 55, 93, 29, 84, 12, 98, 1, 63, 86, 43, 60, 47, 82, 35, 80, 13, 6,
     69, 54, 73, 102, 10, 108, 38, 81, 44, 56, 94, 9, 90, 41, 20, 4, 16, 39, 14, 50, 31, 70, 92, 74, 65, 88, 5, 101, 79,
     58, 62, 61, 72, 85, 96, 8, 83, 71, 78, 87, 27, 42, 49, 36, 66, 95, 52, 34, 32, 46, 22, 19, 51, 100, 23, 75, 28, 25,
     15, 2, 97, 40, 77, 11, 67, 104, 21, 0, 64, 53, 17, 107, 48, 37, 59, 103, 106, 30, 109, 99, 54]

def town7_opt4():
    return [47,2,3,15]

def town7_opt5():
    return [78,103,106,36, 18, 0, 21]

def town07_straight():
    return [56, 31]

def town07_right():
    return [56, 42]

def town07_left():
    return [56, 37]

def town7_opt6():
    # Benchmark 2km
    return [56,21,56,42,56,37,108,67,32,76,81]

def town7_opt4km():
    # Benchmark 4km
    return [56,21,56,42,56,37,108,67,32,76,81,78 ,103,106,5,98,104,87,74,25,24]

def town1_opt1():
    # benchmark 2.5km
    return [84, 63, 57, 72, 84, 111, 176, 45, 75, 188, 204, 187, 87, 32, 50]

def town1_opt8km():
    # 4km equvalent:
    # [79,195, 198, 18, 46, 73, 94, 184, 38, 174, 44, 177, 133, 87]
    # benchmark 8km
    return [84, 63, 57, 72, 84, 111, 176, 45, 75, 188, 204, 187, 87, 32, 50, 57, 35, 67, 79, 195, 198, 18, 46, 73, 94,
             184, 38, 174, 44, 177, 133, 87, 79,195, 198, 18, 46, 73, 94,184, 38, 174, 44, 177, 133, 87]

def town2_opt4km():
    # Benchmark 4km
    return [0,18,34,45,63,70,88,84,2,24,28,29,17,13,24,57,73,81,83,44,33,41,64,54,20,11,47,79,72,63,70,82,44,33,35,45,63,50,29,41,65,87,62]

def town2_opt3():
    return [72, 83, 11, 31, 34, 64, 32, 87, 66, 43, 47, 80, 26, 49, 17, 6, 37, 61, 23, 89, 59, 35, 19, 40, 86, 63, 41, 1, 12,
     20, 62, 22, 65, 10, 76, 46, 84, 2, 50, 79, 85, 21, 4, 30, 36, 53, 25, 77, 69, 3, 8, 81, 88, 0, 7, 51, 5, 73, 56,
     44, 74, 13, 39, 52, 42, 70, 78, 48, 90, 45, 24, 57, 27, 60, 58, 38, 33, 54, 29, 28, 71, 18, 82, 15, 9, 55, 16, 67,
     14, 68, 75]


def town15_opt1():
    return [112,66,25,104,160,167,212,284]

def all_spawn_points(spawn_points):
    return [i for i in range(len(spawn_points))]