import time
import numpy as np
import matplotlib.pyplot as plt
import packetreader as pr
import copy

# -------------------overhead-------------------
E = np.array(list())  # echoes
S = np.array(list())  # sends
X = list()  # samples for MMD algorithm
t = 0.3  # threshold value that determines whether a new cluster is made
p = 0  # number of cluster centers found
C = list()  # cluster centers
r = list()  # cluster sizes
u = np.zeros((15, 15))
d = int
sequence = []
clusters_needed = True
sends = 0
echoes = 0


class RTT():
    time = 0.0

    def __init__(self, send, echo):
        self.send, self.echo = send, echo
        self.time = self.echo - self.send
        self.time *= 1000

    def __lt__(self, other):
        return True if float(self) < float(other) else False

    def __gt__(self, other):
        return True if float(self) > float(other) else False

    def __eq__(self, other):
        return True if float(self) == float(other) else False

    def __float__(self):
        return self.time

    def __sub__(self, other):
        return float(self) - float(other)

    def __add__(self, other):
        return float(self) + float(other)

    def __repr__(self):
        return str(self.time)


def flatten(listx):
    temp = list()
    for a in listx:
        for b in a:
            temp.append(copy.copy(b))
    for a in range(len(temp)):
        try:
            temp.pop(temp.index(np.nan))
        except:
            pass
    return temp


def get_alpha(C):
    undivided_alpha = 0
    n = 0
    C_length = len(C)
    for i in range(C_length):
        for j in range(C_length):
            if (i != j):
                undivided_alpha += abs(float(C[i]) - float(C[j]))
                n += 1
    #print(alpha)
    return undivided_alpha / n


#print(get_alpha([1, 2, 3, 4, 5, 6, 7, 8]))  # this is 3. so evidently, a/alpha is not the averge distance


def get_element_of_X(i):  # returns inputted number of x; elsewise, returns 0
    try:
        return X[i]
    except:
        return 0


get_element_of_X_vec = np.vectorize(get_element_of_X)


def get_mean_of_cluster(ndarrayx):
    subtotal = 0
    for a in ndarrayx:
        subtotal += float(a)
    return subtotal / np.count_nonzero(ndarrayx)


get_mean_of_cluster_vec = np.vectorize(get_mean_of_cluster)

# -------------loading in packet data--------------
with open("Data/SeparateLocations/3-connection-dataset1.txt", "r") as packetData:
    first_line = packetData.readline()
    first_time = pr.get_time(first_line)
    send_source = pr.get_source(first_line)
    S = np.append(S, pr.get_time(first_line))
    for line in packetData.readlines()[:-2]:
        # print(end="")     #dummy line
        timeStamp = "at " + str(pr.get_time(line))
        if pr.check_push_flag(line):
            if pr.check_if_send(line, send_source):
                # print("send from", pr.get_source(line), "to", pr.get_destination(line), timeStamp, "\n")
                S = np.append(S, pr.get_time(line) - first_time)
            else:
                # print("echo from", pr.get_source(line), "to", pr.get_destination(line), timeStamp, "\n")
                E = np.append(E, pr.get_time(line) - first_time)
print(E)
print(S)
difference_limit = 6
differences = [[np.nan for i in range(difference_limit)] for j in range(S.size)]
for a in range(S.size):
    difference_limiter = 0
    for b in range(E.size):
        if E[b] - S[a] > 0 and difference_limiter < difference_limit:
            differences[a][difference_limiter] = RTT(S[a], E[b])
            #print("updating differences[" + str(b) + "][" + str(difference_limiter) + "] to", RTT(S[a], E[b]))
            difference_limiter += 1
print(differences)

# differences = np.where(differences < 0.5, differences, 0)
# for a in differences:
#     for b in a:
#         print("here be", b.time)

# ------------------debug output------------------
print(len(E), "Echoes and", len(S), "Sends mapped to array 'differences' of shape", (S.size, difference_limit))

# Starting a timer to measure the runspeed of the MMD algorithm
start_time = time.time()

# ------------------MMD prototype------------------
X = flatten(differences)
# X = list(np.concatenate((np.random.poisson(50, 100), np.random.poisson(100, 100), np.random.poisson(150, 50), np.random.poisson(250, 100), np.random.poisson(600, 40))))
# print("X is", X)
j0 = int
alpha = 0
X_prime = X.copy()
print("\n\nX' is", X_prime, "\n\n")
# print("\nBefore MMD:")
# print("inputted X:\t", X, "\nX':\t", X_prime, "\nC:\t", C, "\np:\t", p, "\nu:\t", u, "\nalpha:\t", alpha)
# setting first cluster center to first item in sample set
x1 = X_prime.index(min(X_prime))
print("x1:", x1)
C.append(X_prime.pop(x1))

# setting first element of u to 1
u[0][0] = x1

# setting C2 to j0-th element of x, s.t. the j0-th element of X is the element with the greatest distance from C1,
# which should be the largest element of X, since there are no negative elements in the set and C1 is the first (and
# thus by the nature of X also the smallest) element of X. also, setting second element of u to j0
j0 = X_prime.index(max(X_prime))
u[0][1] = X.index(X_prime[j0])
C.append(X_prime.pop(j0))

# updating number of clusters found
p = 2

# updating alpha
alpha = get_alpha(C)

# outputting data after initial step
# print("\nAfter first step of MMD:")
# print("inputted X:\t", X, "\nX':\t", X_prime, "\nC:\t", C, "\np:\t", p, "\nu:\t", u, "\nalpha:\t", alpha)

# checking if there are elements of x that are too far from pre-existing clusters
print("Creating clusters...")
while clusters_needed:
    distances = {}
    minimum_distances = {}
    #print(len(X_prime))
    for a in range(len(X_prime)):
        distances = {}
        for b in range(p):
            distances.update({abs(X_prime[a] - C[b]): X_prime[a]})
        minimum_distances.update({min(distances.keys()): distances[min(distances.keys())]})
    d = max(minimum_distances.keys())
    x_i0 = minimum_distances[d]
    if d < t * alpha:
        clusters_needed = False
    else:
        p += 1
        C.append(X_prime.pop(X_prime.index(x_i0)))
        u[0][p - 1] = X.index(x_i0)
        alpha = get_alpha(C)
    print("clusters:", C, "\ndistances:", len(distances), "\nd:", d, "\nalpha:", alpha, "\nclusters needed:", clusters_needed, "\n")

# initializing r for 1 <= j <= p
r = [1] * p

# appropriately shrinking u to the size of its contained data
u_0 = u[0]
u = np.empty((len(X), p)) * np.nan
for a in range(len(u[0])):
    u[0][a] = u_0[a]
# matching elements in X' to their closest elements of C via u
print("Partitioning data into determined clusters...")
for i in range(len(X_prime)):
    j = 0
    for b in range(len(C)):
        if abs(X_prime[i] - C[b]) < abs(X_prime[i] - C[j]):
            j = b
    r[j] += 1
    u[r[j] - 1][j] = X.index(X_prime[i])
# outputting data after clustering
# print("inputted X:\t", X, "\nX':\t", X_prime, "\nC:\t", C, "\np:\t", p, "\nu:\t", u, "\nr:\t", r, "\nalpha:\t", alpha)

# updating cluster centers to be means of the clusters
print("Updating cluster centers...")
clusters = get_element_of_X_vec(u.astype(int))
print(clusters)
for j in range(len(C)):
    C[j] = get_mean_of_cluster(clusters[:, j])

# Calculating runtime
end_time = time.time()

# ------------------final output-------------------
print("\n\n\n\n\n\n\nThe Specifics:")
print("\n\nClusters:\n", clusters)
print("\n\nCluster Centers:\t", C)
print("\n\nMMD execution time:\t", end_time - start_time, "seconds")

# ---visualizing the inputted data for reference---
#plt.hist(X, density=True)  # This is debug code; for visualizing input
#plt.show()
