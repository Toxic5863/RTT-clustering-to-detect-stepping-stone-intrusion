import time
import numpy as np
import matplotlib.pyplot as plt
import packetreader as pr
import copy
import statistics
import os

# -------------------overhead-------------------
E = list()  # echoes
S = list()  # sends
X = list()  # samples for MMD algorithm
t = 0.02  # threshold value that determines whether a new cluster is made
p = 0  # number of cluster centers found
C = list()  # cluster centers
r = list()  # cluster sizes
u = np.zeros((1000, 1000))
d = int
average_clustering_ratio = int
number_of_clustering_ratios = 0
high_ratio_clusters = list()
clusters_needed = True
sends = 0
echoes = 0


class RTT():
    time = 0.0

    def __init__(self, send, echo):
        self.send, self.echo = send, echo
        self.time = self.echo - self.send
        # self.time *= 1000

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
    print("Flattening", listx)
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
    # print(alpha)
    return undivided_alpha / n


def cluster_analysis_update():
    print("\nNo. of Clusters:" + str(p))
    print("\nShape of clusters:", np.shape(clusters))
    print("\nCluster sizes are:")
    print(" | ", end="")
    for a in range(p):
        print(np.count_nonzero(clusters[:, a]), end=" | ")
    print("\n\n\n")


# print(get_alpha([1, 2, 3, 4, 5, 6, 7, 8]))  # this is 3. so evidently, a/alpha is not the averge distance


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
with open("Data/verification.txt", "r") as packetData:
    first_line = packetData.readline()
    # first_time = pr.get_time(first_line)
    send_source = pr.get_source(first_line)
    S.append(pr.get_time(first_line))
    print("source:", send_source)
    # print(packetData.readlines())
    # time.sleep(10)
    for line in packetData.readlines()[:]:
        # print(end="")     #dummy line
        timeStamp = "at " + str(pr.get_time(line))
        if pr.check_push_flag(line):
            if pr.check_if_send(line, send_source):
                # print("send from", pr.get_source(line), "to", pr.get_destination(line), timeStamp, "\n")
                S.append(pr.get_time(line))
            else:
                # print("echo from", pr.get_source(line), "to", pr.get_destination(line), timeStamp, "\n")
                E.append(pr.get_time(line))

print("Writing send and echo lists to fs...")
with open("sendsfile.txt", "w") as sends_file:
    for a in range(len(S)):
        entry = str(S[a]) + "\t" + str(a) + "\n"
        sends_file.write(entry)

with open("echoesfile.txt", "w") as echoes_file:
    for a in range(len(E)):
        entry = str(E[a]) + "\t" + str(a) + "\n"
        echoes_file.write(entry)
print("Done writing\n")

# print(E)
# print(S)
difference_limit = int(3 * (len(E) / len(S)))
difference_limit = 3
print("Window size:", difference_limit)
differences = [[np.nan for i in range(difference_limit)] for j in range(len(S))]
for a in range(len(S)):
    difference_limiter = 0
    for b in range(len(E)):
        # print(S[a],"and", E[b], end=" ")
        if E[b] - S[a] > 0 and difference_limiter < difference_limit:
            # print("match!")
            differences[a][difference_limiter] = RTT(S[a], E[b])
            # print("updating differences[" + str(b) + "][" + str(difference_limiter) + "] to", RTT(S[a], E[b]))
            difference_limiter += 1
        # else:
        #     # print("don't match!")
# print(differences)


# ------------------debug output------------------
print(len(E), "Echoes and", len(S), "Sends mapped to array 'differences' of shape", (len(S), difference_limit))

# Starting a timer to measure the runspeed of the MMD algorithm
start_time = time.time()

# ------------------MMD prototype------------------
X = flatten(differences)
# X = list(np.concatenate((np.random.poisson(50, 100), np.random.poisson(100, 100), np.random.poisson(150, 50), np.random.poisson(250, 100), np.random.poisson(600, 40))))
j0 = int
alpha = 0
X_prime = X.copy()
print("Length of X':", len(X_prime))
print("X':", X_prime)
# print("\n\nX' is", X_prime, "\n\n")
# print("\nBefore MMD:")
x1 = X_prime.index(min(X_prime))
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
    if not X_prime:
        clusters_needed = False
    # print("clusters:", C, "\ndistances:", len(distances), "\nd:", d, "\nalpha:", alpha, "\nclusters needed:", clusters_needed, "\n")

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
for j in range(len(C)):
    C[j] = get_mean_of_cluster(clusters[:, j])

# Calculating runtime
end_time = time.time()

# ------------------final output-------------------
print("\n\n\n\n\n\n\nThe Specifics:")
# print("\n\nClusters:\n", clusters)
print("\n\nCluster Centers:\t", C)
print("\n\nMMD execution time:\t", end_time - start_time, "seconds")

# ---visualizing the inputted data for reference---
# plt.hist(X, density=True)  # This is debug code; for visualizing input
# plt.show()

# ------------------END OF MMD-------------------
print("\n\n----------END OF MMD---------\n\n", )

# assert len(clusters[0]) == p
cluster_analysis_update()

print("Removing RTTs with duplicate sends...\n")
# removing RTTs from clusters that have duplicate sends
for a in range(len(clusters[0, :])):
    # print("\nChecking cluster", a, "(length:", str(np.count_nonzero(clusters[:, a])) + ") for duplicates")
    new_cluster = list()
    for b in range(np.count_nonzero(clusters[:, a])):
        duplicate = False
        for c in range(len(new_cluster)):
            if clusters[b][a].send == new_cluster[c].send:
                duplicate = True
                new_cluster[c] = min(clusters[b][a], new_cluster[c])
        if not duplicate:
            new_cluster.append(clusters[b][a])
    clusters[:, a] = 0
    clusters[:len(new_cluster), a] = new_cluster

cluster_analysis_update()

print("Removing RTTs with duplicate echoes...\n")
# removing RTTs from clusters that have duplicate echoes
for a in range(len(clusters[0, :])):
    # print("\nChecking cluster", a, "(length:", str(np.count_nonzero(clusters[:, a])) + ") for duplicates")
    new_cluster = list()
    for b in range(np.count_nonzero(clusters[:, a])):
        duplicate = False
        for c in range(len(new_cluster)):
            if clusters[b][a].echo == new_cluster[c].echo:
                duplicate = True
                new_cluster[c] = min(clusters[b][a], new_cluster[c])
        if not duplicate:
            new_cluster.append(clusters[b][a])
    clusters[:, a] = 0
    clusters[:len(new_cluster), a] = new_cluster

cluster_analysis_update()

print("Creating subsets of consecutive elements in clusters...\n")
g = 2
clustering_ratios = dict()
for a in range(len(clusters[0, :])):
    previous_send_index = S.index(clusters[0][a].send)
    consecutive_elements = 0
    send_indices = list()
    for b in range(np.count_nonzero(clusters[:, a])):
        current_RTT = clusters[b][a]
        current_send_index = S.index(current_RTT.send)
        send_indices.append(current_send_index)
        if abs(current_send_index - previous_send_index) <= g:
            consecutive_elements += 1
        previous_send_index = current_send_index
    cluster_range = max(send_indices) - min(send_indices)
    clustering_ratio = (consecutive_elements / cluster_range) if cluster_range != 0 else 1
    clustering_ratios.update({clustering_ratio: a})
    # print("number of elements in cluster", :", np.count_nonzero(clusters[:, a]))
    # print("number of connected elements:", consecutive_elements)

print("\n\nCalculating average clustering ratio...\n")
number_of_clustering_ratios, average_clustering_ratio = 0, 0
for a in list(clustering_ratios.keys()):
    average_clustering_ratio += a
    number_of_clustering_ratios += 1
average_clustering_ratio /= number_of_clustering_ratios

minimum_difference = 1 * statistics.stdev(clustering_ratios)
print("Filtering for clusters whose ratio is two standard deviations above the mean...")
for a in list(clustering_ratios.keys()):
    if a - average_clustering_ratio >= minimum_difference:
        high_ratio_clusters.append(clustering_ratios[a])

# find maximum disjoint subset
# code goes here

# print("\ndebug:")
# last_cluster = clusters[:np.count_nonzero(clusters[:, -1]), -1]
# for a in range(len(last_cluster)):
#     print(S.index(last_cluster[a].send))
print("list of clustering ratios:", clustering_ratios)
print("mean clustering ratio:", average_clustering_ratio)
print("clustering ratios two standard deviations above the mean:", len(high_ratio_clusters))

try:
    os.mkdir("High Ratio Clusters")
except FileExistsError:
    print("Folder 'High Ratio Clusters' already exists")

filename = "highratiocluster"
filenumber = 0
for a in high_ratio_clusters:
    # print(u[np.count_nonzero(u[:, a])])
    with open("High Ratio Clusters/" + filename + str(filenumber) + ".txt", "w") as writefile:
        for b in range(np.count_nonzero(clusters[:, a])):
            entry = "(" + str(clusters[b, a]) + ", " + str(clusters[b, a].send) + ", " + str(clusters[b, a].echo) + ") "
            writefile.write(entry)
    filenumber += 1

try:
    os.mkdir("Clusters")
except FileExistsError:
    print("Folder 'Clusters' already exists")

filename = "clusters"
filenumber = 0
for a in range(np.count_nonzero(clusters[0, :])):
    with open("Clusters/" + filename + str(filenumber) + ".txt", "w") as writefile:
        # for b in range(np.count_nonzero)
        for b in range(np.count_nonzero(clusters[:, a])):
            entry = "(" + str(clusters[b, a]) + ", " + str(clusters[b, a].send) + ", " + str(clusters[b, a].echo) + ") "
            writefile.write(entry)
    filenumber += 1
time.sleep(1)
