import numpy as np

## ---------------------Decision Tree----------------------
# ----------Node Class----------------
class Node():
    """ 
    Node class for Leaf Nodes and Decision Nodes

    --------Features----------
    feature_index : Feature
    threshold : threshold of decision
    left : left child
    right : right child
    var_red : Varriance Reduction
    Value : Only valid for Leaf Node
    """
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, var_red = None, value = None):
        self.feature_index  = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        self.value = value

# ----------------Tree Class-------------
class DecisionTreeRegressor():
    def __init__(self, min_samples_split = 2, max_depth = 2):
        """ -------------Constructor-------------

        root : starting point of the tree
        min_samples_split : stopping criteria for tree
        max_depth :  stopping criteria for tree

        """
        self.root = None
        self.min_samples_split  = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        """
        -------------Recursive function------------
        Split the data and check if the stopping criteria is met.
        If it is not met then we find the best split for the current data
        Use the get_best_split function.
        If the variance reduction by split >0, then we split the dataset
        recursively call  build_tree  function to build left and right subtree


        """
        X, Y = dataset[:,:-1], dataset[:,-1]             # separate the features and target
        num_samples, num_features = np.shape(X)
        best_split = {}
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:          # check if the stoppping criteria is met
            best_split = self.get_best_split(dataset, num_samples, num_features)        # call the get_best_split function
            if best_split["var_red"]>0:                   # if variance reduction > 0 then split the tree
                # recursilvely call left_subtree and right_subtree
                left_subtree  =self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["var_red"])
            
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)
    


    def calculate_leaf_value(self, Y):
        """
        Calculate the mean of all the values belonging to a particular leaf node
        """
        val = np.mean(Y)
        return val
    


    def get_best_split(self, dataset, num_samples, num_features):
        """
        Find the best split.
        Define a Dictionary to store the best split.
        Initialize maximum variance reduction as negative infinity
        1st for loop:
            loop over all the features using the index location
        2nd for loop:
            Loop over all the feature values
            split the data between left and right based on the threshold
        1st if loop:
            check if the  child is not null
            compute thte information gain (reduction in variance)
        2nd if loop:
            Update the best_split if current variance reduction > max variance reduction.
        return best split
        """
        best_split = {}
        max_var_red = -float("inf")
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)                       # all the unique values in the feature/column
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:,-1]
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    if curr_var_red>max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
        return best_split
    

    def variance_reduction(self, parent, l_child, r_child):
        """
        function to find the information gain/ reduction in variance
        """
        weight_l = len(l_child)/len(parent)
        weight_r = len(r_child)/len(parent)
        reduction = np.var(parent) - (weight_l*np.var(l_child) + weight_r*np.var(r_child))
        return reduction
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def print_tree(self, tree = None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent+indent)
            print("%sright" % (indent), end="")
            self.print_tree(tree.right, indent+indent)
    
    def fit(self, X, Y):
        dataset = np.concatenate((X,Y), axis=1)
        self.root = self.build_tree(dataset)

    def make_prediction(self, x, tree):
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction (x, tree.right)
        
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
