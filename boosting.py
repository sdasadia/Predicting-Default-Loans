import numpy as np
from math import log
from math import exp

class decision_tree:
    
    print('Following methods are avilable')
    print('1. get_numpy_data(data, features, output)')
    print('2. intermediate_node_weighted_mistakes(labels_in_node, data_weights)')
    print('3. best_splitting_feature(data, features, target, data_weights)')
    print('4. create_leaf(target_values, data_weights)')
    print('5. weighted_decision_tree_create(data, features, target, data_weights, current_depth = 1, max_depth = 10)')
    print('6. count_nodes(tree)')
    print('7. classify(tree, x, annotate = False)')
    print('8. adaboost_with_tree_stumps(data, features, target, num_tree_stumps)')
    
    def get_numpy_data(self, data, features, output):
        
        """
        Input: data (dataframe), features (list of features), output(output feature)
        Returns feature matrix, output matrix
        
        """
    
        features_array = np.array(data[features])
        output_array = np.array(data[output])
        return(features_array, output_array)
    
    def intermediate_node_weighted_mistakes(self, labels_in_node, data_weights):
        
        """
        Input: labels_in_node (classifying label (+1 / -1), data_weights (weights for each data point)
        returns: weight of mistakes
            
        """
        # Sum the weights of all entries with label +1
        total_weight_positive = sum(data_weights[labels_in_node == +1])
    
        # Weight of mistakes for predicting all -1's is equal to the sum above
        weighted_mistakes_all_negative = total_weight_positive
    
        # Sum the weights of all entries with label -1
        total_weight_negative = sum(data_weights[labels_in_node == -1])
    
        # Weight of mistakes for predicting all +1's is equal to the sum above
        weighted_mistakes_all_positive = total_weight_negative
    
        # Return the tuple (weight, class_label) representing the lower of the two weights
        # class_label should be an integer of value +1 or -1.
        # If the two weights are identical, return (weighted_mistakes_all_positive,+1)
        if weighted_mistakes_all_positive <= weighted_mistakes_all_negative:
            return (weighted_mistakes_all_positive, +1)
        else:
            return (weighted_mistakes_all_negative, -1)


    def best_splitting_feature(self, data, features, target, data_weights):
        
        """
        Input: data, features (features used to predict target), target, data_weights (weights)
        Returns: best_feature (which feature to split on -- based on minimum classification error on training data)
         
        """
    
        # These variables will keep track of the best feature and the corresponding error
        best_feature = None
        best_error = float('+inf')
        num_points = float(len(data))
    
        # Loop through each feature to consider splitting on that feature
        for feature in features:
        
            # The left split will have all data points where the feature value is 0
            # The right split will have all data points where the feature value is 1
            left_split = data[data[feature] == 0]
            right_split = data[data[feature] == 1]
        
            # Apply the same filtering to data_weights to create left_data_weights, right_data_weights
            left_data_weights = data_weights[data[feature] == 0]
            right_data_weights = data_weights[data[feature] == 1]
        
            # Calculate the weight of mistakes for left and right sides
            left_weighted_mistakes, left_class = intermediate_node_weighted_mistakes(left_split[target], left_data_weights)
            right_weighted_mistakes, right_class = intermediate_node_weighted_mistakes(right_split[target], right_data_weights)
        
            # Compute weighted error by computing
            #  ( [weight of mistakes (left)] + [weight of mistakes (right)] ) / [total weight of all data points]
            error = (left_weighted_mistakes + right_weighted_mistakes)/(sum(left_data_weights) + sum(right_data_weights))
        
            # If this is the best error we have found so far, store the feature and the error
            if error < best_error:
                best_feature = feature
                best_error = error
    
            # Return the best feature we found
            return best_feature

    def create_leaf(self, target_values, data_weights):
        
        """
        Function that creates a leaf (a datastructure to represent a tree
            
        """
    
        # Create a leaf node
        leaf = {'splitting_feature' : None,
            'is_leaf': True}
    
        # Computed weight of mistakes.
        weighted_error, best_class = intermediate_node_weighted_mistakes(target_values, data_weights)

        # Store the predicted class (1 or -1) in leaf['prediction']
        leaf['prediction'] = best_class
    
        return leaf


    def weighted_decision_tree_create(self, data, features, target, data_weights, current_depth = 1, max_depth = 10):
        
        """
        Function to learn weighted decision tree
        Returns: leaf and nodes for each split
            
        """
        
        remaining_features = features[:] # Make a copy of the features.
        target_values = data[target]
        print "--------------------------------------------------------------------"
        print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    
        # Stopping condition 1. Error is 0.
        if intermediate_node_weighted_mistakes(target_values, data_weights)[0] <= 1e-15:
            print "Stopping condition 1 reached."
            return create_leaf(target_values, data_weights)

        # Stopping condition 2. No more features.
        if remaining_features == []:
            print "Stopping condition 2 reached."
            return create_leaf(target_values, data_weights)
    
        # Additional stopping condition (limit tree depth)
        if current_depth > max_depth:
            print "Reached maximum depth. Stopping for now."
            return create_leaf(target_values, data_weights)

        splitting_feature = best_splitting_feature(data, features, target, data_weights)
        remaining_features.remove(splitting_feature)
    
        left_split = data[data[splitting_feature] == 0]
        right_split = data[data[splitting_feature] == 1]
    
        left_data_weights = data_weights[data[splitting_feature] == 0]
        right_data_weights = data_weights[data[splitting_feature] == 1]
    
        print "Split on feature %s. (%s, %s)" % (\
                                             splitting_feature, len(left_split), len(right_split))
        
        # Create a leaf node if the split is "perfect"
        if len(left_split) == len(data):
            print "Creating leaf node."
            return create_leaf(left_split[target], data_weights)
        if len(right_split) == len(data):
            print "Creating leaf node."
            return create_leaf(right_split[target], data_weights)
                                                             
        # Repeat (recurse) on left and right subtrees
        left_tree = weighted_decision_tree_create(
             left_split, remaining_features, target, left_data_weights, current_depth + 1, max_depth)
        right_tree = weighted_decision_tree_create(
             right_split, remaining_features, target, right_data_weights, current_depth + 1, max_depth)
                                                                     
        return {'is_leaf'          : False,
                'prediction'       : None,
                'splitting_feature': splitting_feature,
                'left'             : left_tree,
                'right'            : right_tree}

    def count_nodes(self, tree):
        
        """
        count no of tree
        """
        
        if tree['is_leaf']:
            return 1
        return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])

    def classify(self,tree, x, annotate = False):
        
        """
        Function that classifies one data point
        Input: Tree data structure created in weighted_decision_tree_create
    
        """
        # If the node is a leaf node.
        if tree['is_leaf']:
            if annotate:
                print "At leaf, predicting %s" % tree['prediction']
            return tree['prediction']
        else:
            # Split on feature.
            split_feature_value = x[tree['splitting_feature']]
            if annotate:
                print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
            if split_feature_value == 0:
                return classify(tree['left'], x, annotate)
            else:
                return classify(tree['right'], x, annotate)


    def adaboost_with_tree_stumps(self, data, features, target, num_tree_stumps):
        
        """
        Implement Adaboost with decision tree
        Returns decision tree and weights (for each data point)
        
        """
        # start with unweighted data
        alpha = graphlab.SArray([1.]*len(data))
        weights = []
        tree_stumps = []
        target_values = data[target]
    
        for t in xrange(num_tree_stumps):
            print '====================================================='
            print 'Adaboost Iteration %d' % t
            print '====================================================='
            # Learn a weighted decision tree stump. Use max_depth=1
            tree_stump = weighted_decision_tree_create(data, features, target, data_weights=alpha, max_depth=1)
            tree_stumps.append(tree_stump)
        
            # Make predictions
            predictions = data.apply(lambda x: classify(tree_stump, x))
        
            # Produce a Boolean array indicating whether
            # each data point was correctly classified
            is_correct = predictions == target_values
            is_wrong   = predictions != target_values
        
            # Compute weighted error
            weighted_error = sum(alpha[is_wrong])/sum(alpha)
        
            # Compute model coefficient using weighted error
            weight = 1./2. * log((1 - weighted_error)/weighted_error)
            weights.append(weight)
        
            # Adjust weights on data point
            adjustment = is_correct.apply(lambda is_correct : exp(-weight) if is_correct else exp(weight))
        
            # Scale alpha by multiplying by adjustment
            # Then normalize data points weights
            alpha = (alpha * adjustment)/float(sum(alpha))
    
        return weights, tree_stumps
