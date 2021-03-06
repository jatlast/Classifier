########################################################################
# Jason Baumbach
#   
#       Classifier:
#           Plugin Linear Discriminant Function (LDF)
#
# Note: this code is available on GitHub 
#   https://github.com/jatlast/Classifier/
#   sub directory LDF/LDF.py
#
########################################################################

#   -- LDF specific --
# get the inner product (dot product) of two equal length vectors
def GetInnerProductOfTwoVectors(vOne, vTwo):
    product = 0
    v_one_len = len(vOne)
    v_two_len = len(vTwo)
    # vOne & vTwo must be of equal length
    if(v_one_len != v_two_len):
        print(f"Warning DP: {v_one_len} != {v_two_len}")
        return -1
    else:
        for i in range(0, v_one_len):
            product += float(vOne[i]) * float(vTwo[i])
    return product

#   -- LDF specific --
# calculate the target type means from the training data
def AddTargetTypeMeansToVarDict(dTrainingData, dVariables):
    col_sums_dic = {} # key = target | value = sum of column values
    row_count_dic = {} # key = target | value = number of rows to divide by

    # zero out the col_sums and row_count dictionaries
    for key in dVariables['target_types']:
        col_sums_dic[key] = {} # col_sums_dic[target][col_name] = sums by target
        row_count_dic[key] = 0 # row_count_dic[target][col_name] = rows by target
        # dynamically create target mean vectors for each target type to the variables dictionary for LDF calculations
        dVariables[key] = {'ldf_mean' : []} # initialized to the empty list
        # loop thought the sared attributes list
        for col in dVariables['shared_attributes']:
            if col != dVariables['target_col_name']:
                # initialize the column sum to zero since this is a shared attribute column
                col_sums_dic[key][col] = 0

    # Loop through the training set to calculate the totals required to calculate the means
    for i in range(1, len(dTrainingData) - 1): # loop through the traing set rows
        for j in range(0, len(dTrainingData[0])): # loop through the traing set columns
            for col in dVariables['shared_attributes']: # loop through the shared columns
                # check if the column is shared
                if dTrainingData[0][j] == col:
                    # only sum the non-target columns
                    if col != dVariables['target_col_name']:
                        # sum the colum values
                        col_sums_dic[dTrainingData[i][dVariables['target_col_index_train']]][col] += float(dTrainingData[i][j])
                    # use the target column as a que to increment the row count
                    else:
                        # incrament the row count
                        row_count_dic[dTrainingData[i][dVariables['target_col_index_train']]] += 1

    # dynamically calculate the appropriate number of target means
    for key in dVariables['target_types']: # loop through the target types
        for col in col_sums_dic[key]: # loop through the columns that were summed by target
            # debug info
            if dVariables['verbosity'] > 2:
                print(f"col:{col}:\t{col_sums_dic[key][col]} / {row_count_dic[key]}")
            # append the colum mean to the target type mean vector
            if row_count_dic[key] > 0:
                dVariables[key]['ldf_mean'].append(col_sums_dic[key][col] / row_count_dic[key])
            else:
                # this should never happen
                print(f"Warning: LDF mean = 0 for target:{key}")
                print(f"col:{col}:\t{col_sums_dic[key][col]} / {row_count_dic[key]}")
                dVariables[key]['ldf_mean'].append(0)

#   -- LDF specific --
# calculate the largets and second largets g(x) to determine the target and confidence of the LDF function
def AddCalculatesOfPluginLDFToVarDic(vTestData, dVariables):
    # initialize calculation variables
    dVariables['ldf_best_g'] = -1 # use -1 so best begins < least possible g(d)
    dVariables['ldf_second_best_g'] = -1 # use -1 so second best begins < least possible g(d)
    dVariables['ldf_best_target'] = 'UNK'
    dVariables['ldf_second_best_target'] = 'UNK'
    dVariables['ldf_confidence'] = 0
    ldf_diff = 0
    # loop through the target types
    for key in dVariables['target_types']:
        # calculate the inner (dot) products of the target type means
        dVariables[key]['ldf_dot_mean'] = GetInnerProductOfTwoVectors(vTestData, dVariables[key]['ldf_mean'])
        # calculate g(x)
        dVariables[key]['ldf_g'] = (2 * dVariables[key]['ldf_dot_mean']) - dVariables[key]['ldf_mean_square']

        # store the largest and second largest g(x) for later comparison to determine confidence
        # current better than second best
        if dVariables['ldf_second_best_g'] < dVariables[key]['ldf_g']:
            # current better than best
            if dVariables['ldf_best_g'] < dVariables[key]['ldf_g']:
                # set best to current
                dVariables['ldf_best_g'] = dVariables[key]['ldf_g']
                dVariables['ldf_best_target'] = key
            else:
                # set second best to current best
                dVariables['ldf_second_best_g'] = dVariables[key]['ldf_g']
                dVariables['ldf_second_best_target'] = key
        # current better than best
        elif dVariables['ldf_best_g'] < dVariables[key]['ldf_g']:
            # set second best to previous best
            dVariables['ldf_second_best_g'] = dVariables['ldf_best_g']
            dVariables['ldf_second_best_target'] = dVariables['ldf_best_target']
            # set best to current
            dVariables['ldf_best_g'] = dVariables[key]['ldf_g']
            dVariables['ldf_best_target'] = key

        # debug info: print the formul used
        if dVariables['verbosity'] > 2:
            print(f"\t{key}_g(x): {round(dVariables[key]['ldf_g'], 2)} = (2 * {round(dVariables[key]['ldf_dot_mean'], 2)}) - {round(dVariables[key]['ldf_mean_square'], 2)}")

    # get the difference between best and second best
    ldf_diff = dVariables['ldf_best_g'] - dVariables['ldf_second_best_g']

    # reset the max
    if dVariables['ldf_diff_max'] < ldf_diff:
        dVariables['ldf_diff_max'] = ldf_diff
    
    # reset the min
    if dVariables['ldf_diff_min'] > ldf_diff:
        dVariables['ldf_diff_min'] = ldf_diff
    
    # use min-max to calculate confidence if min & max have been initialized
    if dVariables['ldf_diff_max'] != dVariables['ldf_diff_min']:
        dVariables['ldf_confidence'] = ((ldf_diff - dVariables['ldf_diff_min']) / (dVariables['ldf_diff_max'] - dVariables['ldf_diff_min']))
    else:
        dVariables['ldf_confidence'] = ldf_diff

    # debugging: sum all LDF confidenc <= 0
    if dVariables['ldf_confidence'] < 0:
        dVariables['ldf_confidence_zero'] += 1
        if dVariables['verbosity'] > 2:
            print(f"ldf diff:{dVariables['ldf_best_g']} - {dVariables['ldf_second_best_g']}")

#   -- LDF specific --
# calculate the inner (dot) products of the different target type means
def AddMeanSqCalcsToVarDic(dVariables):
    # loop through the target types
    for key in dVariables['target_types']:
        # calculate the inner (dot) products of the different target type means
        dVariables[key]['ldf_mean_square'] = GetInnerProductOfTwoVectors(dVariables[key]['ldf_mean'], dVariables[key]['ldf_mean'])
