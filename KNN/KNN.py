########################################################################
# Jason Baumbach
#   
#       Classifier:
#           K-Nearest Neighbor (KNN)
#
# Note: this code is available on GitHub
#   https://github.com/jatlast/Classifier/
#   sub directory KNN/KNN.py
#
########################################################################
print("KNN imported")
#   -- KNN specific --
# create a dictionary of list objects equal to the number of neighbors to test
def InitNeighborsDict(neighbors_dict, k):
    for i in range(1, k + 1):
        neighbors_dict[i] = {'index' : -1, 'distance' : 1000, 'type' : ''}

#   -- KNN specific --
# compute Euclidean distance between any two given vectors with any length.
def EuclideanDistanceBetweenTwoVectors(vOne, vTwo):
    distance = 0
    v_one_len = len(vOne)
    v_two_len = len(vTwo)
    # vOne & vTwo must be of equal length
    if(v_one_len != v_two_len):
        print(f"Warning UD: {v_one_len} != {v_two_len}")
        return -1

    for p in range(0, v_one_len):
        distance += math.pow((abs(float(vOne[p]) - float(vTwo[p]))), 2)
    return math.sqrt(distance)

#   -- KNN specific --
# Populate the k-nearest neighbors by comparing all training data with test data point
def PopulateNearestNeighborsDicOfIndexes(dTrainingData, vTestData, dVariables):
    distances = []  # for debugging only (store then sort all distances for comparison to the chosen distances)
    # Loop through the training set (sans header) to find the least distance(s)
    for i in range(1, len(dTrainingData) - 1):
        # create the training vector of only the shared attributes to compare to the passed in testing vector (vTestData)
        train_vec = GetVectorOfOnlySharedAttributes(dTrainingData, i, dVariables)
        # get the Euclidean distance between the test & train vectors
        EuclideanDistance = EuclideanDistanceBetweenTwoVectors(vTestData, train_vec)
        distances.append(EuclideanDistance) # for debugging only
        # reset neighbor tracking variables
        neighbor_max_index = -1 # index of neighbor furthest away
        neighbor_max_value = -1 # value of neighbor furthest away
        # Loop through the neighbors dict so the maximum stored is always replaced first
        for j in range(1, len(dVariables['neighbors_dict']) + 1):
            if dVariables['neighbors_dict'][j]['distance'] > neighbor_max_value:
                neighbor_max_value = dVariables['neighbors_dict'][j]['distance']
                neighbor_max_index = j
        # save the newest least distance over the greatest existing neighbor distance
        # compare the current Euclidean distance against the value of neighbor furthest away
        if EuclideanDistance < neighbor_max_value:
            # since current distance is less, replace neighbor with max distance with current distance info
            dVariables['neighbors_dict'][neighbor_max_index]['index'] = i
            dVariables['neighbors_dict'][neighbor_max_index]['distance'] = EuclideanDistance
            dVariables['neighbors_dict'][neighbor_max_index]['type'] = dTrainingData[i][dVariables['target_col_index_train']]

    # debugging: print least k-distances from all k-distances calculated for comparison with chosen neighbors
    if dVariables['verbosity'] > 2:
        distances.sort()
        print("least distances:")
        for i in range(0, len(dVariables['neighbors_dict'])):
            print(f"min{i}:({distances[i]}) \t& neighbors:({dVariables['neighbors_dict'][i+1]['distance']})")

#   -- KNN specific --
# Calculate majority type and confidence from the majority of nearest neighbors
def AddKNNMajorityTypeToVarDict(dVariables):
    type_count_dict = {} # store key = type & value = sum of neighbors with this type
    # zero out KNN majority type tracking variables
    dVariables['knn_majority_type'] = 'UNK'
    dVariables['knn_majority_count'] = 0
    dVariables['knn_confidence'] = 0
    # loop through the target types and zero out the type_count_dict
    for key in dVariables['target_types']:
        type_count_dict[key] = 0

    # loop through the nearest neighbors and total the different type hits
    for i in range(1, len(dVariables['neighbors_dict']) + 1):
        type_count_dict[dVariables['neighbors_dict'][i]['type']] += 1

    # loop through the target types to set the majority info for KNN confidence calculation
    for key in type_count_dict:
        # current is better than best
        if dVariables['knn_majority_count'] < type_count_dict[key]:
            # set best to current
            dVariables['knn_majority_count'] = type_count_dict[key]
            dVariables['knn_majority_type'] = key

    # calculate confidence as (majority / k-nearest neighbors)
    dVariables['knn_confidence'] = dVariables['knn_majority_count'] / len(dVariables['neighbors_dict'])

    # debug info
    if dVariables['verbosity'] > 2:
        print(f"majority:{dVariables['knn_majority_type']}{type_count_dict}")

