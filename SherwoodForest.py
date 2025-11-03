# RandomSherwood.py
import numpy as np
from RF import DecisionTreeClassifierScratch  # din trädklass i RF.py

class RandomSherwoodForestClassifier:
    """
    En enkel Random Forest-bygge ovanpå ditt egna beslutsträd.
    - mode='bootstrap'  -> klassisk RF (med återläggning)
    - mode='bagging'    -> utan återläggning (använder bag_fraction)
    - oob_score=True    -> beräknar out-of-bag-accuracy
    """
    def __init__(self,
                 n_estimators=100,
                 max_depth=20,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_impurity_decrease=0.0,
                 max_features='sqrt',
                 mode='bootstrap',          # 'bootstrap' eller 'bagging'
                 bag_fraction=0.7,          # används om mode='bagging'
                 random_state=None,
                 oob_score=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.mode = mode
        self.bag_fraction = bag_fraction
        self.random_state = random_state
        self.oob_score = oob_score

        self.bootstrap = (mode != 'bagging')
        self.rng_ = np.random.default_rng(random_state)

        self.trees_ = []
        self.oob_score_ = None

    # -------- helpers --------
    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf in (None, "sqrt"):
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(mf, float):            # andel av features
            return max(1, int(mf * n_features))
        if isinstance(mf, int):              # exakt antal
            return max(1, min(mf, n_features))
        # fallback -> sqrt
        return max(1, int(np.sqrt(n_features)))

    def _bootstrap_indices(self, n_samples: int):
        """Med återläggning (klassisk RF)."""
        idx = self.rng_.integers(0, n_samples, size=n_samples)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[idx] = False
        return idx, oob_mask

    def _bagging_indices(self, n_samples: int):
        """Utan återläggning, tar en fraktion av datan."""
        m = max(1, int(self.bag_fraction * n_samples))
        idx = self.rng_.choice(n_samples, size=m, replace=False)
        # i bagging har vi inte naturlig OOB (kan definieras som resten)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[idx] = False
        return idx, oob_mask

    # -------- API --------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32, order="C")
        y = np.asarray(y).ravel().astype(int)
        n_samples, n_features = X.shape
        max_feats = self._resolve_max_features(n_features)

        # För OOB-beräkning
        oob_votes = None
        if self.oob_score:
            oob_votes = {i: [] for i in range(n_samples)}

        self.trees_.clear()

        for _ in range(self.n_estimators):
            # välj index för detta träd
            if self.bootstrap:
                idx, oob_mask = self._bootstrap_indices(n_samples)
            else:
                idx, oob_mask = self._bagging_indices(n_samples)

            X_sub, y_sub = X[idx], y[idx]

            # Unikt frö till varje träd (men reproducerbart pga rng_)
            seed = int(self.rng_.integers(0, 2**31 - 1))

            # Träna ett träd med feature-subsampling per nod
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=max_feats,          # viktigt: subsampling per nod
                random_state=seed
            )
            tree.fit(X_sub, y_sub)
            self.trees_.append(tree)

            # Samla OOB-prediktioner
            if self.oob_score and oob_mask.any():
                preds = tree.predict(X[oob_mask])
                for j, p in zip(np.where(oob_mask)[0], preds):
                    oob_votes[j].append(p)

        # Räkna OOB-score
        if self.oob_score:
            has_vote = np.array([len(oob_votes[i]) > 0 for i in range(n_samples)])
            if has_vote.any():
                y_oob_pred = np.empty(n_samples, dtype=int)
                for i in np.where(has_vote)[0]:
                    vals, cnts = np.unique(oob_votes[i], return_counts=True)
                    y_oob_pred[i] = vals[np.argmax(cnts)]
                self.oob_score_ = (y_oob_pred[has_vote] == y[has_vote]).mean()
            else:
                self.oob_score_ = None

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32, order="C")
        # (n_estimators, n_samples)
        all_preds = np.vstack([t.predict(X) for t in self.trees_])
        n_samples = X.shape[0]
        y_pred = np.empty(n_samples, dtype=all_preds.dtype)
        for i in range(n_samples):
            vals, cnts = np.unique(all_preds[:, i], return_counts=True)
            y_pred[i] = vals[np.argmax(cnts)]
        return y_pred

    # (valfritt) enkel predict_proba via frekvens
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32, order="C")
        all_preds = np.vstack([t.predict(X) for t in self.trees_])  # (T, N)
        classes = np.unique(all_preds)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        proba = np.zeros((X.shape[0], len(classes)), dtype=np.float32)
        for i in range(X.shape[0]):
            vals, cnts = np.unique(all_preds[:, i], return_counts=True)
            for v, c in zip(vals, cnts):
                proba[i, class_to_idx[v]] = c
        proba /= self.n_estimators
        return proba, classes
