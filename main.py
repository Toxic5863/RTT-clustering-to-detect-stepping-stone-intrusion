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
t = 0.001  # threshold value that determines whether a new cluster is made
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


class RTT(): # this object stores round trip times and their associated packets.
    time = 0.0 # it functions as a float for the sake of operations

    def __init__(self, send, echo): # each RTT object is a structure of a send packet, echo packet, and the consequent round trip time of the two
        self.send, self.echo = send, echo
        self.time = self.echo - self.send

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


def flatten(listx):  # this takes the two-dimensional RTT data and flattens it to one dimension to be fed to the MMD algorithm
    print("Flattening", listx)
    temp = list()  # creating a new list in which the "flattened" data will be stored.
    for a in listx:
        for b in a:
            temp.append(copy.copy(b)) # adding items from each subset of the 2d array to temp
    for a in range(len(temp)):
        try:
            temp.pop(temp.index(np.nan)) # removing non-number (non-RTT) data from temp
        except:
            pass
    return temp


def get_alpha(C): # This is a in the MMD algorithm, which is multipled by t to set a threshold for creating new clusters
    undivided_alpha = 0
    n = 0
    C_length = len(C)
    for i in range(C_length):           # for each cluster, determine its distances with each other cluster
        for j in range(C_length):       # and calculate the average of all of these distances
            if (i != j):
                undivided_alpha += abs(float(C[i]) - float(C[j]))
                n += 1
    # print(alpha)
    return undivided_alpha / n


def cluster_analysis_update(): # This function is used to output the current status of the clusters, for seeing how they change after each step.
    print("\nNo. of Clusters:" + str(p))
    print("\nShape of clusters:", np.shape(clusters))
    print("\nCluster sizes are:")
    print(" | ", end="")
    for a in range(p):
        print(np.count_nonzero(clusters[:, a]), end=" | ")
    print("\n\n\n")


def get_element_of_X(i):  # returns inputted number of x; elsewise, returns 0
    try:
        return X[i]
    except:
        return 0


get_element_of_X_vec = np.vectorize(get_element_of_X)   # vectorizing the function for translating indices to
                                                        # corresponding elements of x for use on u


def get_mean_of_cluster(ndarrayx):
    subtotal = 0
    for a in ndarrayx:                                  # gets the mean of a cluster
        subtotal += float(a)
    return subtotal / np.count_nonzero(ndarrayx)


get_mean_of_cluster_vec = np.vectorize(get_mean_of_cluster) # vectorizes the function so it can be applied to all of the
                                                            # clusters simultaneously

# -------------loading in packet data--------------
with open("Data/CapturedTraffic.txt", "r") as packetData: # using a text file of packets
    first_line = packetData.readline() # getting the first line of the data
    send_source = pr.get_source(first_line) # establishing a source IP for use in packet matching
    S.append(pr.get_time(first_line)) # updating S to include the first line
    print("source:", send_source)
    for line in packetData.readlines()[:]:  # parsing each line in the packet data
        # timeStamp = "at " + str(pr.get_time(line)) # the time and making a string for output
        if pr.check_push_flag(line): # checking to make sure the push flag is raised to avoid acks
            if pr.check_if_send(line, send_source):
                # print("send from", pr.get_source(line), "to", pr.get_destination(line), timeStamp, "\n")
                S.append(pr.get_time(line)) # adding the send to set S
            else:
                # print("echo from", pr.get_source(line), "to", pr.get_destination(line), timeStamp, "\n")
                E.append(pr.get_time(line)) # adding the echo to set E

print("Writing send and echo lists to fs...")
with open("sendsfile.txt", "w") as sends_file: # writing the set S to a file for debugging
    for a in range(len(S)):
        entry = str(S[a]) + "\t" + str(a) + "\n"
        sends_file.write(entry)

with open("echoesfile.txt", "w") as echoes_file: # writing the set E to a file for debugging
    for a in range(len(E)):
        entry = str(E[a]) + "\t" + str(a) + "\n"
        echoes_file.write(entry)
print("Done writing\n")

difference_limit = 3 # this is the window within which we look for echoes after each send
print("Window size:", difference_limit)
differences = [[np.nan for i in range(difference_limit)] for j in range(len(S))] # an empty 2d array in which the potential RTTs will be stored
for a in range(len(S)): # for each send packet
    difference_limiter = 0 # resetting difference_limiter for use as a counter of the number of echoes checked aftera  send
    for b in range(len(E)): #for each echo packet
        if E[b] - S[a] > 0 and difference_limiter < difference_limit:   # if the RTT is positive (meaning the echo comes after the send) and we
                                                                        # have not used 3 subsequent echoes yet
            differences[a][difference_limiter] = RTT(S[a], E[b]) # add the potential RTT to the 2d array
            difference_limiter += 1 # update the number of subsequent echoes used so far


# ------------------debug output------------------
print(len(E), "Echoes and", len(S), "Sends mapped to array 'differences' of shape", (len(S), difference_limit)) # debug output

# Starting a timer to measure the runspeed of the MMD algorithm
start_time = time.time() # checking the start time so that we can time the MMD algorithm

# ------------------MMD prototype------------------
X = flatten(differences)    # flattening the structure "differences" of potential RTTs and using it as the input set for MMD
j0 = int
alpha = 0 # the a value that we use for determining whether to make a new cluster
X_prime = X.copy() # creating X', a copy of X that we will modify throughout the MMD algorithm
print("Length of X':", len(X_prime))
print("X':", X_prime)
x1 = X_prime.index(min(X_prime)) # getting the smallest RTT (which should be the first send minus the first echo
C.append(X_prime.pop(x1)) # using the smallest RTT as our first cluster

# setting first element of u to 1
u[0][0] = x1 # updating u to include our newly created cluster center

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

# checking if there are elements of x that are too far from pre-existing clusters
print("Creating clusters...")
while clusters_needed:
    distances = {}
    minimum_distances = {}
    for a in range(len(X_prime)):
        distances = {}
        for b in range(p):
            distances.update({abs(X_prime[a] - C[b]): X_prime[a]}) # creating a dictionary of all the distances between
                                                                   # element a and each cluster
        minimum_distances.update({min(distances.keys()): distances[min(distances.keys())]}) # making a dictionary of the
                                                                                            # smallest of those distances
    d = max(minimum_distances.keys())   # getting the maximum of the minimum distances
    x_i0 = minimum_distances[d]
    if d < t * alpha:  # checking against a whether to make a new cluster
        clusters_needed = False
    else:
        p += 1 # updating the number of clusters to reflect a new one
        C.append(X_prime.pop(X_prime.index(x_i0))) # adding a new cluster
        u[0][p - 1] = X.index(x_i0) # updating the clusters as represented in u
        alpha = get_alpha(C) # re-calculating a
    if not X_prime:
        clusters_needed = False

# initializing r for 1 <= j <= p
r = [1] * p

# appropriately shrinking u to the size of its contained data
u_0 = u[0]
u = np.empty((len(X), p)) * np.nan   # remaking u to accommodate all of the elements
for a in range(len(u[0])):
    u[0][a] = u_0[a]
# matching elements in X' to their closest elements of C via u
print("Partitioning data into determined clusters...")
for i in range(len(X_prime)):
    j = 0
    for b in range(len(C)):
        if abs(X_prime[i] - C[b]) < abs(X_prime[i] - C[j]): # checking the distance from element i to each cluster
            j = b
    r[j] += 1
    u[r[j] - 1][j] = X.index(X_prime[i]) # updating u to reflect the newly added elements in the cluster

# updating cluster centers to be means of the clusters
print("Updating cluster centers...")
clusters = get_element_of_X_vec(u.astype(int)) # translating the indices in u to their corresponding elements in X
for j in range(len(C)):
    C[j] = get_mean_of_cluster(clusters[:, j]) # calculating the means of each cluster

# Calculating runtime
end_time = time.time() # stopping the clock to check how long the MMD algorithm took

# ------------------final output-------------------
print("\n\n\n\n\n\n\nThe Specifics:")
print("\n\nCluster Centers:\t", C)
print("\n\nMMD execution time:\t", end_time - start_time, "seconds")


# ------------------END OF MMD-------------------
print("\n\n----------END OF MMD---------\n\n", )

# assert len(clusters[0]) == p
cluster_analysis_update()

print("Removing RTTs with duplicate sends...\n")
# removing RTTs from clusters that have duplicate sends
for a in range(len(clusters[0, :])):
    new_cluster = list() # creating a new cluster to which unique elements will be added
    for b in range(np.count_nonzero(clusters[:, a])):
        duplicate = False # resetting duplicate flag
        for c in range(len(new_cluster)):
            if clusters[b][a].send == new_cluster[c].send:
                duplicate = True # raising duplicate flag
                new_cluster[c] = min(clusters[b][a], new_cluster[c]) # if element shares a send with one already in new_cluster, replace
        if not duplicate:                                            # the element in new cluster with the smaller RTT of the two
            new_cluster.append(clusters[b][a]) # adding the element to new_cluster if the duplicate flag was not raised
    clusters[:, a] = 0
    clusters[:len(new_cluster), a] = new_cluster # updating the cluster in 'clusters' to the new_cluster created

cluster_analysis_update()

print("Removing RTTs with duplicate echoes...\n")
# removing RTTs from clusters that have duplicate echoes
for a in range(len(clusters[0, :])):
    new_cluster = list() # creating a new cluster to which unique elements will be added
    for b in range(np.count_nonzero(clusters[:, a])):
        duplicate = False # reset duplicate flag
        for c in range(len(new_cluster)):
            if clusters[b][a].echo == new_cluster[c].echo:
                duplicate = True # raise duplicate flag
                new_cluster[c] = min(clusters[b][a], new_cluster[c]) # if element shares an echo with one already in new_cluster, replace
        if not duplicate:                                            # the element in new cluster with the smaller RTT of the two
            new_cluster.append(clusters[b][a]) # adding the element to new_cluster if the duplicate flag was not raised
    clusters[:, a] = 0
    clusters[:len(new_cluster), a] = new_cluster # updating the cluster in 'clusters' to the new_cluster created

cluster_analysis_update()

print("Creating subsets of consecutive elements in clusters...\n")
g = 2 # the window for determining whether elements are "consecutive"
clustering_ratios = dict() # clustering ratios will be stored in a dictionary tied to their corresponding clusters' indices
for a in range(len(clusters[0, :])):
    previous_send_index = S.index(clusters[0][a].send) # setting the first "previous element" to be the first element in
    consecutive_elements = 0                           # the cluster and setting the number of consecutive elements to 0
    send_indices = list()
    for b in range(np.count_nonzero(clusters[:, a])): # parsing RTTs in the clusters. Non-elements are set to 0
        current_RTT = clusters[b][a]
        current_send_index = S.index(current_RTT.send) # getting the index of the RTT being currently checked
        send_indices.append(current_send_index)        # keeping track of the number of indices checked
        if abs(current_send_index - previous_send_index) <= g: # if the distance between the previous index and the
            consecutive_elements += 1                          # current one is within window g, the consecutive elements
        previous_send_index = current_send_index               # list is updated and the previous index is set to the current one
    cluster_range = len(send_indices)             # the range of the cluster is equal to the number of elements in it
    clustering_ratio = (consecutive_elements / cluster_range) # calculating the clustering ratio
    clustering_ratios.update({a: clustering_ratio}) # adding the clustering ratio and its corresponding cluster index to
                                                    # a dictionary

# removing anomalously small clusters
average_cluster_size = 0  # initializing the average cluster size
cluster_sizes = list()
for a in clustering_ratios.keys():
    cluster_sizes.append(np.count_nonzero(clusters[:, a])) # adding all of the clusters' sizes to a list
average_cluster_size = sum(cluster_sizes) / len(cluster_sizes) # calcuating the average value of that list
new_clustering_ratios = dict()
for a in clustering_ratios.keys():
    if not(np.count_nonzero(clusters[:, a]) < 0.3 * average_cluster_size): # removing clustering ratios corresponding
        new_clustering_ratios.update({a: clustering_ratios[a]}) # to clusters that are too small
clustering_ratios = new_clustering_ratios # updating the clustering ratios to the new ones


print("\n\nCalculating average clustering ratio...\n")
number_of_clustering_ratios, average_clustering_ratio = 0, 0
for a in clustering_ratios.values():
    average_clustering_ratio += a                       # calculating the average clustering ratio by taking the mean of
    number_of_clustering_ratios += 1                    # the clustering_ratios dictionary
average_clustering_ratio /= number_of_clustering_ratios

print(clustering_ratios.values())
clustering_ratio_std_dev = statistics.stdev(clustering_ratios.values()) # getting the standar dev. of the set of clustering ratios
minimum_difference = 0.03 * clustering_ratio_std_dev # calculating the number of standard deviations that
print("Standard deviation:", minimum_difference)     # a cluster should be above the mean to make it into the final set
print("Filtering for clusters whose ratio is two standard deviations above the mean...")
for a in clustering_ratios.keys():
    if clustering_ratios[a] - average_clustering_ratio >= minimum_difference: # filtering out clustering ratios that are
        high_ratio_clusters.append(a)                                         # not the minimum number of std. devs above the mean

# find maximum disjoint subset
# code goes here

print("list of clustering ratios:")
for a in clustering_ratios.keys():
    print("cluster", str(a) + ":", clustering_ratios[a])
print("mean clustering ratio:", average_clustering_ratio)
print("clustering ratios two standard deviations above the mean:", len(high_ratio_clusters))

try:
    os.mkdir("Selected Clusters") # making a folder to put the resulting clusters' data in
except FileExistsError:
    print("Folder 'Selected Clusters' already exists")

filename = "selected cluster "
filenumber = 0
for a in high_ratio_clusters:
    with open("Selected Clusters/" + filename + str(filenumber) + ".txt", "w") as writefile:
        ordered_cluster = list()
        for b in range(np.count_nonzero(clusters[:, a])): # writing the clusters' data to the files
            ordered_cluster.append(clusters[b, a])
        for b in sorted(ordered_cluster, key=lambda x: x.send):
            entry = "(" + str(b) + ", " + str(S.index(b.send)) + ", " + str(E.index(b.echo)) + ") "
            writefile.write(entry)
    filenumber += 1

try:
    os.mkdir("Clusters") # making a folder to put the clusters' data in
except FileExistsError:
    print("Folder 'Clusters' already exists")

filename = "cluster "
filenumber = 0
for a in range(np.count_nonzero(clusters[0, :])):
    with open("Clusters/" + filename + str(filenumber) + ".txt", "w") as writefile:
        ordered_cluster = list()
        for b in range(np.count_nonzero(clusters[:, a])):
            ordered_cluster.append(clusters[b, a]) # adding the clusters to a list that will be ordered
        for b in sorted(ordered_cluster, key=lambda x: x.send): # ordering the clusters by send packet
            entry = "(" + str(b)+ ", " + str(S.index(b.send)) + ", " + str(E.index(b.echo)) + ") " # writing the clusters
            writefile.write(entry)                                                                 # to a file
    filenumber += 1
time.sleep(1)
