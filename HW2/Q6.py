import resource
import datetime

'''
This is an inefficient recursive program to explore all possible number of weights given the contraint 
that the input dimension is `INPUT` and the total number of hidden neurons are `N_HIDDEN`

ex.1 INPUT = 10, N_HIDDEN = 36
Maximum # weights: 510, Network structure [10, 22, 14, 1]
Minumun # weights: 46, Network structure [10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Total possibilities: 9227465
Peak memory usage: 9244KB
User time: 0:03:10.492064s

ex.2 INPUT = 12, N_HIDDEN = 48
Maximum # weights: 877, Network structure [12, 29, 19, 1]
Minumun # weights: 60, Network structure [12, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Total possibilities: 2971215073
Peak memory usage: 9180KB
User time: 16:42:28.524118s
'''

N_HIDDEN = 36
INPUT = 10

# Q6
count = 0
maxi = 0
mini = 1000
result_max = []
result_min = []


def subset_sum(numbers, target, partial=[]):

    s = sum(partial)

    # check if the partial sum is equals to target
    if s == target:
        global result_max
        global result_min
        global count
        count += 1
        #print("Generate hidden structure", partial)
        calculate_weight(partial, result_max, result_min)
        return
    if s >= target:
        return  # if we reach the number why bother to continue

    for i in range(len(numbers)):
        n = numbers[i]
        subset_sum(numbers, target, partial+[n])


def calculate_weight(hidden, result_max, result_min):
    wnum = 0
    hidden.append(1)  # OUTPUT
    print("Explore network structure", [INPUT] + hidden, end=' ')
    for j in range(len(hidden)-1):
        if j != len(hidden)-1-1:
            wnum += hidden[j] * (hidden[j+1] - 1)
        else:
            wnum += hidden[j] * (hidden[j+1])
    wnum += INPUT * (hidden[0]-1)
    print("# of weights %d" % wnum)
    global maxi
    global mini
    if wnum > maxi:
        result_max.clear()
        maxi = wnum
        result_max.append(str([INPUT] + hidden))
    elif wnum == maxi:
        result_max.append(str([INPUT] + hidden))
    if wnum < mini:
        result_max.clear()
        mini = wnum
        result_min.append(str([INPUT] + hidden))
    elif wnum == mini:
        result_min.append(str([INPUT] + hidden))


if __name__ == '__main__':
    subset_sum([i+2 for i in range(N_HIDDEN-1)],
               N_HIDDEN)  # [2, 3 ... N_HIDDEN]

    for i in range(len(result_max)):
        print('Maximum # weights: ', maxi, ', Network structure', result_max[i])
    for i in range(len(result_min)):
        print('Minimum # weights: ', mini, ', Network structure', result_min[i])
    print("Explore %d possibilities" % count)

    print('Peak memory usage = %dKB' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print('User time = %ss' % str(datetime.timedelta(
        seconds=float(resource.getrusage(resource.RUSAGE_SELF).ru_utime))))
