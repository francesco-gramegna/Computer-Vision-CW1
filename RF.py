import numpy as np
# Node class
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Feature index for splitting. Could be height of a person
        self.threshold = threshold      # Threshold value for splitting. If height < threshold  (could be 70kg or somethin) go left
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.value = value              # Class label for leaf nodes

    def is_leaf(self):
        
        return self.value is not None

def make_leaf(y):
    # Help function to create leaf nodes
    # Make a leaf node by finding the majority class in y
    # Return the class label as node value
    unique_classes, counts = np.unique(y, return_counts=True)
    score = 0
    for i in range(len(unique_classes)):
        if counts[i] > score:
            score = counts[i]
            biggest_class = unique_classes[i]
        
    return Node(value=biggest_class)

def predict_leaf(x, node: Node):
    # Traverse the tree make prediction for input x
    while not node.is_leaf():
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

# function for impurity: GINI
# Gini Impurity: Measures how often a randomly chosen element from the set would be incorrectly labeled if randomly assigned a label according to the distribution of labels in the node. It's fast to compute and widely used in classification tasks.
# Gathered from https://www.displayr.com/how-is-splitting-decided-for-decision-trees/
def gini_impurity(y):   
    i = np.unique(y)
    p_tot = 0
    for class_label in i:
        p_i = np.sum(y == class_label) / len(y)
        p_tot += + p_i**2

    return 1 - p_tot

# Test gini function
#y = np.array([0, 0, 1, 1, 1])
#print(gini_impurity(y))

# Logic for best split:
# Goal: F(feature, threshold) = argmax Information Gain
# IG = Gini(parent) - ( y_left/ y_total * Gini(left) + y_right/y_total * Gini(right) )
def best_splitold(X, y, min_samples_leaf=1):
    y= np.asarray(y).ravel() # 1d array pooper

    bestGini = float('inf')
    bestGain  = 0
    bestFeature = None
    bestThreshold = None
    bestLeftIndices = None
    bestRightIndices = None
    nImages, nFeatures = X.shape # X is 520 x 2576

    parentGini = gini_impurity(y)

    for feature in range(nFeatures):  
        col = X[:, feature] # Whole column for that feature
        thresholds = splitInHalf(col)

        for threshold in thresholds:
            leftIndices = col <= threshold # Go left if value less than threshold
            rightIndices = col > threshold # Go right if value greater than threshold

            nLeft = leftIndices.sum() 
            nRight = rightIndices.sum()

            if nLeft == 0 or nRight == 0:
                continue
            
            #print(y)

            if nLeft < min_samples_leaf or nRight < min_samples_leaf:
                continue
            #print("Y left:", y[leftIndices])
            giniLeft = gini_impurity(y[leftIndices])
            giniRight = gini_impurity(y[rightIndices])
            weightedGini = (nLeft / nImages) * giniLeft + (nRight / nImages) * giniRight
            IG = parentGini - weightedGini


            # If this was the best gain so far --> Store it
            if IG > bestGain:
                bestGain = IG
                bestGini = weightedGini
                bestFeature = feature
                bestThreshold = threshold
                bestLeftIndices = np.where(leftIndices)[0] # Dont know why np.where need to be used here but yeah
                bestRightIndices = np.where(rightIndices)[0]


    #print(y[leftIndices])
    return bestFeature, bestThreshold, bestGain, bestLeftIndices, bestRightIndices

# Made by ChatGPT, improved version of best_split above. First one was too slow.
def best_split(X, y, min_samples_leaf=1, max_features=None, rng=None):
    """
    Snabb och robust best_split:
    - Feature-subsampling per nod via max_features (int/None)
    - Overflow-säker threshold: x1 + (x2 - x1) * 0.5
    - Skannar sorterad kolumn och uppdaterar klassräkningar inkrementellt
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_samples, n_features = X.shape

    if rng is None:
        rng = np.random.default_rng(0)

    # vilka features testas?
    feats = np.arange(n_features)
    if max_features is not None and max_features < n_features:
        feats = rng.choice(feats, size=max_features, replace=False)

    # label-encoding av y för snabbare bincount
    classes, y_enc = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    parent_gini = gini_impurity(y)
    best_gain = 0.0
    best_feature = None
    best_threshold = None
    best_left_idx = None
    best_right_idx = None

    for f in feats:
        # sortera kolumnen; aritmetik i float64 för stabilitet
        order = np.argsort(X[:, f], kind='mergesort')
        xf = X[order, f].astype(np.float64, copy=False)
        yf = y_enc[order]

        # hoppa över feature med alla lika värden
        if xf[0] == xf[-1]:
            continue

        # om icke-finit förekommer: ersätt med median (sällsynt men säkert)
        if not np.isfinite(xf).all():
            finite = np.isfinite(xf)
            if not finite.any():
                continue
            med = np.median(xf[finite])
            xf[~finite] = med

        # alla börjar till höger
        right_counts = np.bincount(yf, minlength=n_classes).astype(np.int64)
        left_counts = np.zeros(n_classes, dtype=np.int64)

        left_n = 0
        # skanna potentiella snitt mellan i och i+1
        for i in range(n_samples - 1):
            c = yf[i]
            left_counts[c] += 1
            right_counts[c] -= 1
            left_n += 1
            right_n = n_samples - left_n

            # respektera minsta bladstorlek
            if left_n < min_samples_leaf or right_n < min_samples_leaf:
                continue
            # endast snitt när värdena faktiskt skiljer sig
            if xf[i] == xf[i + 1]:
                continue

            # gini(left/right)
            pL = left_counts / left_n
            gL = 1.0 - np.sum(pL * pL)

            pR = right_counts / right_n
            gR = 1.0 - np.sum(pR * pR)

            weighted = (left_n / n_samples) * gL + (right_n / n_samples) * gR
            gain = parent_gini - weighted
            if gain > best_gain:
                best_gain = gain
                # overflow-säker mittpunkt
                x1 = xf[i]
                x2 = xf[i + 1]
                thr = x1 + (x2 - x1) * 0.5
                best_feature = f
                best_threshold = thr

                # lagra index i originalordning
                mask_left = X[:, f] <= thr
                best_left_idx = np.where(mask_left)[0]
                best_right_idx = np.where(~mask_left)[0]

    return best_feature, best_threshold, best_gain, best_left_idx, best_right_idx


# To get middle of uniqeu threshhold values
# Helpfunction to best_split
def splitInHalf(vector):
    uniqueValues = np.unique(vector)
    if len(uniqueValues) <= 1: # Special case if we are at the leaf
        return []
    return (uniqueValues[:-1] + uniqueValues[1:]) / 2 # Split in half between unique values

# Build tree check
def buildTreeOld(X, y, depth=0, max_depth=5, min_samples_split=10, min_samples_leaf=5, min_impurity_decrease=0.0):
    # min_sapmples_leaf allow to adjust the leaf size. To avoid overfitting
    # min_impurity_decrease: If the best split does not decrease impurity by at least this amount, dont split. Prevents unnecessary splits (computatio)
    X = np.asarray(X)
    y = np.asarray(y).ravel()  # Gör y till 1D array
    nImages, nFeatures = X.shape

    # Stopping criteria
    # 1: Max depth reached
    # 2: Not enough samples to split
    # 3: All samples belong to the same class
    if depth >= max_depth or nImages < min_samples_split or len(np.unique(y)) == 1:
        return make_leaf(y)

    # Find the best split
    feature, threshold, gain, left_indices, right_indices = best_split(X, y, min_samples_leaf=min_samples_leaf)

    # If no valid split found, make a leaf
    if feature is None:
        return make_leaf(y)
    
    # min_samples_leaf implemention:
    if left_indices.size < min_samples_leaf or right_indices.size < min_samples_leaf:
        return make_leaf(y)
    
    # min_impurity_decrease implementation:
    if gain <= min_impurity_decrease:
        return make_leaf(y)

    # Recursively build left and right subtrees
    left_node = buildTree(X[left_indices], y[left_indices], depth + 1, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease)
    right_node = buildTree(X[right_indices], y[right_indices], depth + 1, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease)
    #Bomba
    return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

def buildTree(X, y,
              depth=0,
              max_depth=5,
              min_samples_split=10,
              min_samples_leaf=5,
              min_impurity_decrease=0.0,
              max_features=None,           # NYTT: feature-subsampling per nod
              rng=None):                   # NYTT: determinism / vidare till best_split
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    nImages, _ = X.shape

    # Stoppvillkor
    if (depth >= max_depth or
        nImages < min_samples_split or
        len(np.unique(y)) == 1):
        return make_leaf(y)

    # Hitta bästa splitten (NYTT: skickar max_features & rng)
    feature, threshold, gain, left_indices, right_indices = best_split(
        X, y,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        rng=rng
    )

    # Om ingen giltig split → blad
    if feature is None:
        return make_leaf(y)

    # Respektera min leaf-storlek & min impurity decrease
    if left_indices.size < min_samples_leaf or right_indices.size < min_samples_leaf:
        return make_leaf(y)
    if gain <= min_impurity_decrease:
        return make_leaf(y)

    # Bygg rekursivt (NYTT: skicka vidare max_features & rng)
    left_node = buildTree(X[left_indices], y[left_indices],
                          depth=depth + 1,
                          max_depth=max_depth,
                          min_samples_split=min_samples_split,
                          min_samples_leaf=min_samples_leaf,
                          min_impurity_decrease=min_impurity_decrease,
                          max_features=max_features,
                          rng=rng)

    right_node = buildTree(X[right_indices], y[right_indices],
                           depth=depth + 1,
                           max_depth=max_depth,
                           min_samples_split=min_samples_split,
                           min_samples_leaf=min_samples_leaf,
                           min_impurity_decrease=min_impurity_decrease,
                           max_features=max_features,
                           rng=rng)

    return Node(feature=feature, threshold=threshold,
                left=left_node, right=right_node)

class DecisionTreeClassifierScratchOld:
    def __init__(self, max_depth=5, min_samples_split = 1, min_samples_leaf=1, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root_ = None

    # Sklearn-lik API
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()  # Gör y till 1D array
        # Anropar din buildTree (se till att den tar samma kwargs)
        self.root_ = buildTree(
            X, y,
            depth=0,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease
        )
        return self
    def predict(self, X):
        X = np.asarray(X)
        predictions = [predict_leaf(x, self.root_) for x in X]
        return np.array(predictions)
    
class DecisionTreeClassifierScratch:
    def __init__(self,
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_impurity_decrease=0.0,
                 max_features=None,          # NYTT: subsampling per nod
                 random_state=None):          # NYTT: determinism
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        self.rng_ = np.random.default_rng(random_state)
        self.root_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        # defaulta till sqrt(#features) om ej satt (klassisk RF)
        mf = self.max_features
        if mf is None:
            mf = int(np.sqrt(X.shape[1]))

        self.root_ = buildTree(
            X, y,
            depth=0,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=mf,                # vidare till best_split
            rng=self.rng_                   # vidare till best_split
        )
        return self

    def predict(self, X):
        # Batch-traversering (mycket snabbare än rad-för-rad)
        X = np.asarray(X)
        n = X.shape[0]
        y_pred = np.empty(n, dtype=np.int64)
        stack = [(self.root_, np.arange(n))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf():
                y_pred[idx] = node.value
                continue
            f = node.feature
            thr = node.threshold
            col = X[idx, f]
            left_mask = col <= thr
            if left_mask.any():
                stack.append((node.left, idx[left_mask]))
            if (~left_mask).any():
                stack.append((node.right, idx[~left_mask]))
        return y_pred
# Train and test one tree
# Prints results
def trainOneTree(Xtr, Ytr, Xte, Yte, max_depth=20, min_samples_split=10, min_samples_leaf=5, min_impurity_decrease=0.0):

    # Om X är (features, samples), transponera till (samples, features)
    print("Before:", Xtr.shape, Ytr.shape)
    print("Before:", Xte.shape, Yte.shape)
    if Xtr.shape[0] != len(Ytr) and Xtr.shape[1] == len(Ytr):
        Xtr = Xtr.T
    if Xte.shape[0] != len(Yte) and Xte.shape[1] == len(Yte):
        Xte = Xte.T
    print("After fix:", Xtr.shape, Ytr.shape)  
    print("After fix:", Xte.shape, Yte.shape)  

    # MY TREEE
    clf = DecisionTreeClassifierScratch(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease
        )
    
    # Take time
    time_start = time.process_time()
    print("Training...")
    clf.fit(Xtr, Ytr)
    y_hat_tr = clf.predict(Xtr)
    y_hat_te = clf.predict(Xte)
    print("Done in", (time.process_time() - time_start), "seconds")
    # debug_root(X, y)
    # print_tree_pretty(clf.root_)
    print(f"Settings used: max_depth={clf.max_depth}, min_samples_split={clf.min_samples_split}, min_samples_leaf={clf.min_samples_leaf}, min_impurity_decrease={clf.min_impurity_decrease}")
    print(f"Train acc: { (y_hat_tr == Ytr).mean()*100:1f}%")
    print(f"Test  acc: {(y_hat_te == Yte).mean()*100:1f} %")
    
# =========================# =========================# =========================# =========================# =========================
# TEST FUNCTIONS MADE BY CHATGPT BELOW
# =========================# =========================# =========================# =========================# =========================

def debug_root(X, y):
    print("X shape:", X.shape, " y shape:", y.shape)
    classes, counts = np.unique(y, return_counts=True)
    print("Classes:", classes)
    print("Counts :", counts)
    print("Parent Gini:", gini_impurity(y))

    bf, bt, bg, li, ri = best_split(X, y)
    print("Best feature:", bf)
    print("Best thr    :", bt)
    print("Best gain   :", bg)
    if bf is not None:
        print("Left/Right sizes:", len(li), len(ri))
        # sanity: check masks match
        col = X[:, bf]
        lm = col <= bt
        print("Index match:", np.array_equal(np.where(lm)[0], li))

def print_tree_pretty(node, prefix="", is_left=None):
    """
    Skriv ut trädet med snygga grenar.
    - prefix: byggs upp rekursivt för att rita vertikala linjer korrekt
    - is_left: None för roten, True för vänsterbarn, False för högerbarn
    """
    if node is None:
        print(prefix + "(empty)")
        return

    # Välj gren-symbol för denna nods rad
    connector = ""
    if is_left is True:
        connector = "├── "
    elif is_left is False:
        connector = "└── "

    if node.is_leaf():
        print(prefix + connector + f"Leaf: class={node.value}")
        return

    # Skriv denna nod
    print(prefix + connector + f"[X{node.feature} <= {node.threshold:.3f}]")

    # För barnens prefix:
    # - Om denna nod ritades med "├──", ska efterföljande rader inom samma gren få "│   "
    # - Om denna nod ritades med "└──", ska efterföljande rader få "    "
    # - Roten (is_left is None) beter sig som "gren fortsätter": använd "│   "

    next_prefix_left  = prefix + ("│   " if is_left in (True, None) else "    ")
    next_prefix_right = prefix + ("│   " if is_left in (True, None) else "    ")

    # Rita vänster som vänstergren (├──) och höger som högergren (└──)
    print_tree_pretty(node.left,  next_prefix_left,  True)
    print_tree_pretty(node.right, next_prefix_right, False)

# ========================= MAIN ========================= # ========================= MAIN =========================# 

if __name__ == "__main__":
    import time
    import utils
    import numpy as np
    x_data , y_labels = utils.getImages()
    Xtraining, Xtest, Ytrain, Ytest = utils.separateTrainingTestQ1(x_data, y_labels)
    # Train and test model
    
    X = np.asarray(x_data)
    y = np.asarray(y_labels)
    # had to convert to float32 to avoid sklearn error and ravel to avoid shape error
    Xtr = np.asarray(Xtraining, dtype=np.float32, order="C")
    Xte = np.asarray(Xtest,     dtype=np.float32, order="C")
    Ytr = np.asarray(Ytrain).ravel().astype(int)
    Yte = np.asarray(Ytest).ravel().astype(int)
    
    trainOneTree(Xtr, Ytr, Xte, Yte, max_depth=20, min_samples_split=10, min_samples_leaf=5, min_impurity_decrease=0.0)






   