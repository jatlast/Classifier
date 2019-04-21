########################################################################
# Jason Baumbach
#   
#       Combined Classifier Driver:
#           K-Nearest Neighbor (KNN)
#           Plugin Linear Discriminant Function (LDF)
#
# Note: this code is available on GitHub
#   https://github.com/jatlast/Classifier.git
#
########################################################################

# Import local libraries...
import sys
sys.path.append('C:/Users/jatlast/Google Drive/Code/AI/Classifier/KNN')
import KNN
sys.path.append('C:/Users/jatlast/Google Drive/Code/AI/Classifier/LDF')
import LDF
# import importlib
# moduleKNN = input('KNN.py:')
# importlib.import_module(moduleKNN)
# moduleLDF = input('LDF.py')
# importlib.import_module(moduleLDF)

# required for reading csv files to get just the header
import csv
# required for sqrt function in Euclidean Distance calculation
import math
# required for parsing data files
import re

# allow command line options
import argparse
parser = argparse.ArgumentParser(description="Perform k-Nearest Neighbor, plugin-LDF, and their combination to classify train and test sets of varying n-dimensional data.")
parser.add_argument("-k", "--kneighbors", type=int, choices=range(1, 30), default=3, help="number of nearest neighbors to use")
parser.add_argument("-ft", "--filetrain", default="../uci_hd_preprocessing/data/cleveland_smoke_uci+_normal_train.csv", help="training file name (and path if not in . dir)")
parser.add_argument("-fs", "--filetest", default="../uci_hd_preprocessing/data/cleveland_smoke_uci+_normal_test.csv", help="testing file name (and path if not in . dir)")
parser.add_argument("-tn", "--targetname", default="target", help="the name of the target attribute")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2, 3], default=0, help="increase output verbosity")
args = parser.parse_args()

# variables that are useful to pass around
variables_dict = {
    'training_file' : args.filetrain
    , 'testing_file' : args.filetest
    , 'verbosity' : args.verbosity
    # UCI Heart Disease specific - attribut universally ignored
    #   Note: if they exist they are represented by the pre-processing added "smoke" attribute
    , 'ignore_columns' : ['cigs', 'years']
    # variables to enable dynamic use/ignore of target attribute
    , 'target_col_name' : args.targetname
    , 'target_col_index_train' : 0
    , 'target_col_index_test' : 0
    , 'training_col_count' : 0
    , 'testing_col_count' : 0
    # algorithm variables summed and compared after testing (com = combined)
    , 'test_runs_count' : 0
    , 'com_knn_ldf_right' : 0
    , 'com_knn_right_ldf_wrong' : 0
    , 'com_ldf_right_knn_wrong' : 0
    , 'com_ldf_wrong_knn_right' : 0
    , 'com_knn_wrong_ldf_right' : 0
    , 'com_knn_ldf_wrong' : 0
    # total times LDF confidence = 0
    , 'ldf_confidence_zero' : 0
    # used to compute the min-max of LDF confidence
    , 'ldf_diff_min' : 1000
    , 'ldf_diff_max' : 0
    # initialization of the three confusion matrices
    , 'knn_confusion_matrix' : {
        'TN'    : 0
        , 'FP'  : 0
        , 'FN'  : 0
        , 'TP'  : 0
    }
    , 'ldf_confusion_matrix' : {
        'TN'    : 0
        , 'FP'  : 0
        , 'FN'  : 0
        , 'TP'  : 0
    }
    , 'com_confusion_matrix' : {
        'TN'    : 0
        , 'FP'  : 0
        , 'FN'  : 0
        , 'TP'  : 0
    }
}

# Specific to UCI's Heart Disease data set which has two target columns: num (0-4) & target (0 or 1)
# Allows the code to dynamically ignore the column NOT specified in the command line
if args.targetname == 'num':
    variables_dict['ignore_columns'].append('target')
elif args.targetname == 'target':
    variables_dict['ignore_columns'].append('num')

variables_dict['neighbors_dict'] = {}
KNN.InitNeighborsDict(variables_dict['neighbors_dict'], args.kneighbors)

# debug info
if args.verbosity > 0:
    print(f"neighbors: {args.kneighbors} = {len(variables_dict['neighbors_dict'])} :len(neighbors_dict)")

# Read the command line specified CSV data files
def ReadFileDataIntoDictOfLists(sFileName, dDictOfLists):
    # read the file
    with open(sFileName) as csv_file:
        line_number = 0
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            dDictOfLists[line_number] = row
            line_number += 1

# dynamically determine target index for inclusion/exclusion when creating vectors for comparison
def AddTargetIndexesToVarDict(dDictOfLists, dVariables):
    # column cound can vary between train and test sets
    col_count = len(dDictOfLists[0])
    # save the column count to the specified train or test variable in the dictionary
    if dDictOfLists['type'] == 'training':
        dVariables['training_col_count'] = col_count
    elif dDictOfLists['type'] == 'testing':
        dVariables['testing_col_count'] = col_count
    else:
        # this should never happen
        print(f"Warning: type {dDictOfLists['type']} not recognized.")
    # loop through the header to find and save the target column index to the variables dictionary
    for col in range(0, col_count):
        # check if the column name matches the target name command line option
        if dDictOfLists[0][col] == dVariables['target_col_name']:
            # save the column index of the found target column name
            if dDictOfLists['type'] == 'training':
                dVariables['target_col_index_train'] = col
            elif dDictOfLists['type'] == 'testing':
                dVariables['target_col_index_test'] = col

# dynamically determine target types in the training file
def AddTargetTypesToVarDict(dTrainingData, dVariables):
    dVariables['target_types'] = {} # key = type & value = count
    # loop through the training set (ignoring the header)
    for i in range(1, len(dTrainingData) - 1):
        # check if target type has already been discovered and added to the target_types variable
        if dTrainingData[i][dVariables['target_col_index_train']] not in dVariables['target_types']:
            # set to 1 upon first discovery
            dVariables['target_types'][dTrainingData[i][dVariables['target_col_index_train']]] = 1
        else:
            # otherwise sum like instances
            dVariables['target_types'][dTrainingData[i][dVariables['target_col_index_train']]] += 1

# dynamically determine the attributes shared between the train and test sets
def AddSharedAttributesToVarDict(dTrainingData, dTestingData, dVariables):
    dVariables['shared_attributes'] = [] # list of header names identical across train and test headers
    # check which data set is larger then loop through the smaller on the outside
    if dVariables['training_col_count'] < dVariables['testing_col_count']:
        # the training set is smaller so loop through its header on the outside
        for i in dTrainingData[0]:
            # ignore the irrelevant columns hard-coded at the beginning of the program ("num" or "target" are added dynmacially) 
            if i not in dVariables['ignore_columns']:
                # loop through the testing set header
                for j in dTestingData[0]:
                    # append matching header name to the list of shared attributes
                    if i == j:
                        dVariables['shared_attributes'].append(i)
    # the testing set is smaller...
    else:
        # ...so loop through its header on the outside
        for i in dTestingData[0]:
            # ignore the irrelevant columns...
            if i not in dVariables['ignore_columns']:
                # loop through the training set header
                for j in dTrainingData[0]:
                    # append matching header name to the list of shared attributes
                    if i == j:
                        dVariables['shared_attributes'].append(i)

# create and return a vector of only the shared attributes (excludes target attributes)
def GetVectorOfOnlySharedAttributes(dDictOfLists, sIndex, dVariables):
    header_vec = [] # for debugging only
    return_vec = [] # vector containing only shared attributes of row values at sIndex 
    col_count = 0 # train or test column number
    # set the appropriate col_count by data set type
    if dDictOfLists['type'] == 'training':
        col_count = dVariables['training_col_count']
    elif dDictOfLists['type'] == 'testing':
        col_count = dVariables['testing_col_count']
    else:
        # this should never happen
        print(f"Warning: type {dDictOfLists['type']} not recognized.")
    # loop through the header of the passed in data set
    for i in range(0, col_count):
        # ignore the row data at the index of the training set target attribute
        if i == dVariables['target_col_index_train'] and dDictOfLists['type'] == 'training':
            continue
        # ignore the row data at the index of the testing set target attribute
        elif i == dVariables['target_col_index_test'] and dDictOfLists['type'] == 'testing':
            continue
        # loop through the shared attributes list
        for col in dVariables['shared_attributes']:
        # check if the passed in header name matches a shared attribbute
            if dDictOfLists[0][i] == col:
                # append the shared attribute value at row[sIndex] col[i]
                return_vec.append(dDictOfLists[sIndex][i])
                # store for debugging incorrectly matched attributes
                header_vec.append(col)

    # debugging info
    if args.verbosity > 2:
        print(f"shared {dDictOfLists['type']}:{header_vec}")
    
    return return_vec

# sum and track confusion matrix by sPrefix (i.e., knn, ild, com) and store in the variables dictionary
def TrackConfusionMatrixSums(sTestType, sPredictionType, sPrefix, dVariables):
    # target is positive
    if int(float(sTestType)) > 0: # unfortunately, assuming the target is numeric makes the code dependant on numerical target values
        # target matches prediction
        if sTestType == sPredictionType:
            # increment true positive (TP) count
            dVariables[sPrefix + '_confusion_matrix']['TP'] += 1
        # target does not match prediction
        else:
            # increment false positive (FP) count
            dVariables[sPrefix + '_confusion_matrix']['FP'] += 1
    # target is negative
    else:
        # target matches prediction
        if sTestType == sPredictionType:
            # increment true negative (TN) count
            dVariables[sPrefix + '_confusion_matrix']['TN'] += 1
        # target does not match prediction
        else:
            # increment false negative (FN) count
            dVariables[sPrefix + '_confusion_matrix']['FN'] += 1

# simply print the confusion matrix by sPrefix (i.e., knn, ild, com) along with calculated stats
def PrintConfusionMatrix(sPrefix, dVariables):
    # print the confusion matrix
    print(f"\n{sPrefix} - Confusion Matrix:\n\tTP:{dVariables[sPrefix + '_confusion_matrix']['TP']} | FN:{dVariables[sPrefix + '_confusion_matrix']['FN']}\n\tFP:{dVariables[sPrefix + '_confusion_matrix']['FP']} | TN:{dVariables[sPrefix + '_confusion_matrix']['TN']}")
    # Classifier Accuracy, or recognition rate: percentage of test set tuples that are correctly classified
    #   calculate accuracy = ( (TP + TN) / All )
    dVariables[sPrefix + '_accuracy'] = round((dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['TN']) / (dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['TN'] + dVariables[sPrefix + '_confusion_matrix']['FP'] + dVariables[sPrefix + '_confusion_matrix']['FN']),2)
    # calculate error rate = (1 - accuracy)
    dVariables[sPrefix + '_error_rate'] = round((1 - dVariables[sPrefix + '_accuracy']),2)
    # Sensitivity, or Recall, or True Positive recognition rate (TPR)
    #   Recall: completeness – what % of positive tuples the classifier label positive
    #   Note: 1.0 = Perfect score
    #       calculate sensitivity|recall|TPR = ( TP / (TP + FN) )
    dVariables[sPrefix + '_sensitivity'] = round(dVariables[sPrefix + '_confusion_matrix']['TP'] / (dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['FN']),2)
    # Precision: exactness – what % of tuples labeled positive are positive
    #   calculate precision = ( TP / (TP + FP) )
    dVariables[sPrefix + '_precision'] = round(dVariables[sPrefix + '_confusion_matrix']['TP'] / (dVariables[sPrefix + '_confusion_matrix']['TP'] + dVariables[sPrefix + '_confusion_matrix']['FP']),2)
    # Specificity, or True Negative recognition rate (TNR)
    #   calculate specificity|TNR = ( TN / (TN + FP) )
    dVariables[sPrefix + '_specificity'] = round(dVariables[sPrefix + '_confusion_matrix']['TN'] / (dVariables[sPrefix + '_confusion_matrix']['TN'] + dVariables[sPrefix + '_confusion_matrix']['FP']),2)
    # calculate false positive rate (FPR) = (1 - specificity) or ( FP / (FP + TN) )
    dVariables[sPrefix + '_FPR'] = round(dVariables[sPrefix + '_confusion_matrix']['FP'] / (dVariables[sPrefix + '_confusion_matrix']['FP'] + dVariables[sPrefix + '_confusion_matrix']['TN']),2)
    # F measure (F1 or F-score): harmonic mean of precision and recall
    #   calculate F-score = (2 * precision * recall) / (precision + recall)
    dVariables[sPrefix + '_fscore'] = round((2 * dVariables[sPrefix + '_precision'] * dVariables[sPrefix + '_sensitivity']) / (dVariables[sPrefix + '_precision'] + dVariables[sPrefix + '_sensitivity']),2)
    # print the values calculated above
    print(f"Accuracy   :{dVariables[sPrefix + '_accuracy']}")
    print(f"Error Rate :{dVariables[sPrefix + '_error_rate']}")
    print(f"Sensitivity:{dVariables[sPrefix + '_sensitivity']}")
    print(f"Precision  :{dVariables[sPrefix + '_precision']}")
    print(f"Specificity:{dVariables[sPrefix + '_specificity']}")
    print(f"FPR        :{dVariables[sPrefix + '_FPR']}")
    print(f"F-score    :{dVariables[sPrefix + '_fscore']}")

# keep track of the running totals of which algorithms were correct and/or incorrect
def AddRunningPredictionStatsToVarDict(sTestType, dVariables):
    dVariables['test_runs_count'] += 1
    # COM-bination right
    if sTestType == dVariables['com_best_target']:
        # KNN right
        if sTestType == dVariables['knn_majority_type']:
            # LDF right
            if sTestType == dVariables['ldf_best_target']:
                dVariables['com_knn_ldf_right'] += 1
            # LDF wrong
            else:
                dVariables['com_knn_right_ldf_wrong'] += 1
        # KNN wrong
        else:
            # LDF right
            if sTestType == dVariables['ldf_best_target']:
                dVariables['com_ldf_right_knn_wrong'] += 1
            # LDF wrong
            else:
                # this should never happen!
                print("Warning: COM-bination can never be connrect when both KNN & LDF are incorrect.")
    # COM-bination wrong
    else:
        # KNN right
        if sTestType == dVariables['knn_majority_type']:
            # LDF right
            if sTestType == dVariables['ldf_best_target']:
                # this should never happen!
                print("Warning: COM-bination can never be inconnrect when both KNN & LDF are correct.")
            # LDF wrong
            else:
                dVariables['com_ldf_wrong_knn_right'] += 1
        # KNN wrong
        else:
            # LDF right
            if sTestType == dVariables['ldf_best_target']:
                dVariables['com_knn_wrong_ldf_right'] += 1
            # LDF wrong
            else:
                dVariables['com_knn_ldf_wrong'] += 1

# Load the training data
training_dict = {'type' : 'training'} # set the type for dynamically determining shared attributes
# read the training csv into the training dict
ReadFileDataIntoDictOfLists(variables_dict['training_file'], training_dict)

# add the target indexes of the training set to the variables dictionary
AddTargetIndexesToVarDict(training_dict, variables_dict)

# add the possible target types to the variables dictionary
AddTargetTypesToVarDict(training_dict, variables_dict)
# print debugging info
if args.verbosity > 0:
    for key in variables_dict['target_types']:
        print(f"target types {key}:{variables_dict['target_types'][key]}")

# Load the testing data
testing_dict = {'type' : 'testing'} # set the type for dynamically determining shared attributes
# read the testing csv into the testing dict
ReadFileDataIntoDictOfLists(variables_dict['testing_file'], testing_dict)

# add the target indexes of the testing set to the variables dictionary
AddTargetIndexesToVarDict(testing_dict, variables_dict)

# add the shared attributes for comparing testing data with training data to the variables dictionary
AddSharedAttributesToVarDict(training_dict, testing_dict, variables_dict)
# debugging info
if args.verbosity > 0:
    # shared attribute includes "target"
    print(f"shared attributes:{variables_dict['shared_attributes']}")
    # vector attribute includes all shared attributes except the "target" attribute
    print(f"vector attributes:{GetVectorOfOnlySharedAttributes(testing_dict, 0, variables_dict)}")

#   -- LDF specific --
# add the target type means to the variables dictionary for later use
LDF.AddTargetTypeMeansToVarDict(training_dict, variables_dict)

#   -- LDF specific --
# calculate the inner (dot) products of the different target type means
LDF.AddMeanSqCalcsToVarDic(variables_dict)

# debugging info
if args.verbosity > 1:
    for key in variables_dict['target_types']:
        print(f"{key} mean_sq:{variables_dict[key]['ldf_mean_square']} | mean:{variables_dict[key]['ldf_mean']}")

# debugging info
#if args.verbosity > 0:
# print the train & test shapes (rows: subtract 1 for headers and 1 for starting from zero; cols: subtract 1 for "num" & 1 for "target")
print(f"train: {len(training_dict)-2} x {len(training_dict[0])-2} | test: {len(testing_dict)-2} x {len(testing_dict[0])-2} | shared: {len(variables_dict['shared_attributes'])-1}")

# debugging info
if args.verbosity > 2:
    # Print some of the rows from input files
    print(f"The first 2 training samples with target:{training_dict[0][variables_dict['target_col_index_train']]}:")
    for i in range(0, 2):
        print(f"\t{i} {training_dict[i]}")

    print(f"\nThe first 2 testing samples with target:{testing_dict[0][variables_dict['target_col_index_test']]}")
    for i in range(0, 2):
        print(f"\t{i} {testing_dict[i]}")

# loop through all testing data
for i in range(1, len(testing_dict) - 1):
    # create the test vector at the i-th row from only the shared test & train attributes
    test_vec = GetVectorOfOnlySharedAttributes(testing_dict, i, variables_dict)

    # set the k-nearest neighbors in the neighbors dict
    KNN.PopulateNearestNeighborsDicOfIndexes(training_dict, test_vec, variables_dict)

    # calculate and set the KNN predicted target and confidence in the variables dict
    KNN.AddKNNMajorityTypeToVarDict(variables_dict)

    # calculate and set the LDF predicted target and confidence in the variables dict
    LDF.AddCalculatesOfPluginLDFToVarDic(test_vec, variables_dict)

    # ----- Store the Confusion Matrix running counts -----
    # track KNN confusion matrix running totals
    TrackConfusionMatrixSums(testing_dict[i][variables_dict['target_col_index_test']], variables_dict['knn_majority_type'], 'knn', variables_dict)
    # track LDF confusion matrix running totals
    TrackConfusionMatrixSums(testing_dict[i][variables_dict['target_col_index_test']], variables_dict['ldf_best_target'], 'ldf', variables_dict)

    # combine and store the confusion matrix counts for most confident prediction
    # KNN confidence > LDF confidence
    if variables_dict['knn_confidence'] > variables_dict['ldf_confidence']:
        # set the combined best target = to the KNN predicted target
        variables_dict['com_best_target'] = variables_dict['knn_majority_type']
        # debugging info
        if args.verbosity > 1:
            if variables_dict['ldf_best_target'] != variables_dict['knn_majority_type']:
                print(f"KNN:{variables_dict['knn_majority_type']}>({testing_dict[i][variables_dict['target_col_index_test']]}:confidence) KNN:{variables_dict['knn_majority_type']}:{round(variables_dict['knn_confidence'],2)} | LDF:{variables_dict['ldf_best_target']}:{round(variables_dict['ldf_confidence'],2)}")
    # LDF confidence > KNN confidence
    else:
        # set the combined best target = to the LDF predicted target
        variables_dict['com_best_target'] = variables_dict['ldf_best_target']
        # debugging infoi
        if args.verbosity > 1:
            if variables_dict['ldf_best_target'] != variables_dict['knn_majority_type']:
                print(f"LDF:{variables_dict['com_best_target']}>({testing_dict[i][variables_dict['target_col_index_test']]}:confidence) KNN:{variables_dict['knn_majority_type']}:{round(variables_dict['knn_confidence'],2)} | LDF:{variables_dict['ldf_best_target']}:{round(variables_dict['ldf_confidence'],2)}")

    
    # track Combined confusion matrix running totals
    TrackConfusionMatrixSums(testing_dict[i][variables_dict['target_col_index_test']], variables_dict['com_best_target'], 'com', variables_dict)
    # -----------------------------------------------------

    # add the running totals of predictions for KNN, LDF, & Combined
    AddRunningPredictionStatsToVarDict(testing_dict[i][variables_dict['target_col_index_test']], variables_dict)

    # reset kneighbors_dict
    for i in range(1, args.kneighbors + 1):
        variables_dict['neighbors_dict'][i] = {'index' : -1, 'distance' : 1000, 'type' : ''}

# print the three confusion matrices
PrintConfusionMatrix('knn', variables_dict)
PrintConfusionMatrix('ldf', variables_dict)
PrintConfusionMatrix('com', variables_dict)

# print the prediction stats for KNN, LDF, & Combined
print(f"\nall:      right |                  {variables_dict['com_knn_ldf_right']} \t| {round((variables_dict['com_knn_ldf_right']/variables_dict['test_runs_count']),2)}%")
print(f"com, knn: right | ldf:      wrong: {variables_dict['com_knn_right_ldf_wrong']} \t| {round((variables_dict['com_knn_right_ldf_wrong']/variables_dict['test_runs_count']),2)}%")
print(f"com, ldf: right | knn:      wrong: {variables_dict['com_ldf_right_knn_wrong']} \t| {round((variables_dict['com_ldf_right_knn_wrong']/variables_dict['test_runs_count']),2)}%")
print(f"knn:      right | com, ldf: wrong: {variables_dict['com_ldf_wrong_knn_right']} \t| {round((variables_dict['com_ldf_wrong_knn_right']/variables_dict['test_runs_count']),2)}%")
print(f"ldf:      right | com, knn: wrong: {variables_dict['com_knn_wrong_ldf_right']} \t| {round((variables_dict['com_knn_wrong_ldf_right']/variables_dict['test_runs_count']),2)}%")
print(f"                | all:      wrong: {variables_dict['com_knn_ldf_wrong']} \t| {round((variables_dict['com_knn_ldf_wrong']/variables_dict['test_runs_count']),2)}%")

# print LDF min & max values for reference
print(f"\nldf: min:{round(variables_dict['ldf_diff_min'],2)} | max:{round(variables_dict['ldf_diff_max'],2)}")

# debugging info - print LDF confidence == 0 summation
if variables_dict['ldf_confidence_zero'] > 0:
    print(f"ldf confidence <= 0: {variables_dict['ldf_confidence_zero']} \t| {round((variables_dict['ldf_confidence_zero']/variables_dict['test_runs_count']),2)}%")
