# ====== EVAL-KIT ======
import time
import numpy as np
import utils
from SherwoodForest import RandomSherwoodForestClassifier
import matplotlib.pyplot as plt


# --- 1) Hjälpfunktioner för metriker/utskrifter ---
# --- Plotta confusionsmatris ---
def plot_confusion_matrix(cm, labels, normalize=False, title="Confusion matrix"):
    """
    cm: 2D array (klass x klass)
    labels: ordnade klassetiketter (samma ordning som cm-axlarna)
    normalize: om True visas andelar per rad istället för räkningar
    """
    import numpy as np
    import matplotlib.pyplot as plt

    M = cm.astype(float)
    if normalize:
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        M = M / row_sums

    plt.figure(figsize=(6, 5))
    im = plt.imshow(M, interpolation="nearest")  # låt matplotlib välja standardfärger
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    # Skriv värden i rutorna
    fmt = ".2f" if normalize else "d"
    thresh = M.max() / 2.0 if M.size else 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            #txt = format(val, fmt)
            #plt.text(j, i, round(val, 2), ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def show_image_examples(X, y_true, y_pred, indices, shape=(32, 32)):
    """Visar bilder (t.ex. success/fail) i grid"""
    n = len(indices)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 2, rows * 2))
    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X[idx].reshape(shape), cmap="gray")
        plt.title(f"T:{y_true[idx]} / P:{y_pred[idx]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def show_fail_pairs_with_utils(X, y_true, y_pred, X_ref, y_ref, utils_module, n_pairs=2):
    """
    Shows pairs of misclassified samples using utils.showImages():
    - For each failed test image: the misclassified image (true label)
      and one correctly labeled image from the predicted class.
    """
    import numpy as np

    # find indices where prediction failed
    fails = np.where(y_true != y_pred)[0]
    if len(fails) == 0:
        print("No misclassifications found.")
        return

    fails = fails[:n_pairs]
    images_to_show = []
    titles = []

    for idx in fails:
        true_label = int(y_true[idx+1])
        pred_label = int(y_pred[idx+1])

        # --- Failed image ---
        fail_img = X[idx]
        images_to_show.append(fail_img)
        titles.append(f"True:{true_label} / Pred:{pred_label}")

        # --- Reference image from predicted class ---
        ref_indices = np.where(y_ref == pred_label)[0]
        if len(ref_indices) > 0:
            ref_idx = np.random.choice(ref_indices)
            ref_img = X_ref[ref_idx]
            images_to_show.append(ref_img)
            titles.append(f"Example of class {pred_label}")
        else:
            # fallback if no reference image exists
            images_to_show.append(fail_img)
            titles.append("No ref image found")

    # Convert to correct shape (each image column-wise)
    utils_module.showImages(np.array(images_to_show).T, titles)

def visualize_success_fail(X, y_true, y_pred, utils_module, n_show=8):
    """
    Visar några success och failure samples med utils.showImages() i grid.
    Kräver att utils.showImages() tar (bilder.T, labels_list).
    """
    # Hitta index för success och fail
    successes = np.where(y_true == y_pred)[0][4:n_show+4]
    fails = np.where(y_true != y_pred)[0][4:n_show+4]

    # --- Successes ---
    if len(successes) > 0:
        print(f"\n--- Visualizing {len(successes)} successes ---")
        titles = [f"T:{y_true[i]} / P:{y_pred[i]}" for i in successes]
        utils_module.showImages(X[successes].T, titles)
    else:
        print("\n(No successes found to visualize)")

    # --- Failures ---
    if len(fails) > 0:
        print(f"\n--- Visualizing {len(fails)} failures ---")
        titles = [f"T:{y_true[i]} / P:{y_pred[i]}" for i in fails]
        utils_module.showImages(X[fails].T, titles)
    else:
        print("\n(No failures found to visualize)")

def confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    m = len(labels)
    idx = {lab:i for i,lab in enumerate(labels)}
    cm = np.zeros((m, m), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm, labels

def print_confusion_summary(cm, labels, top_k=10):
    # skriv total korrekt, och topp-förväxlingar
    total = cm.sum()
    correct = np.trace(cm)
    print(f"Confusion matrix: shape={cm.shape}, correct={correct}/{total} ({correct/total:.1%})")
    # mest förväxlade par (off-diagonal)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for i in range(off.shape[0]):
        for j in range(off.shape[1]):
            if off[i, j] > 0:
                pairs.append((off[i, j], labels[i], labels[j]))
    pairs.sort(reverse=True, key=lambda t: t[0])
    if len(pairs) == 0:
        print("No confusions (perfect).")
        return
    print(f"Top {min(top_k, len(pairs))} confusions (true -> predicted : count):")
    for c, ti, pj in pairs[:top_k]:
        print(f"  {ti} -> {pj} : {c}")

def show_example_indices(y_true, y_pred, n=5, kind="fail"):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if kind == "fail":
        idx = np.where(y_true != y_pred)[0]
        title = f"Failures (n={len(idx)})"
    else:
        idx = np.where(y_true == y_pred)[0]
        title = f"Successes (n={len(idx)})"
    idx = idx[:n]
    print(title + " example indices & labels:")
    for k in idx:
        print(f"  idx={k:5d}  true={int(y_true[k])}  pred={int(y_pred[k])}")

# --- 2) En enda körning: träna, mät, skriv ut fint ---
def run_experiment(
    Xtr, Ytr, Xte, Yte,
    n_estimators=5, max_depth=18,
    min_samples_split=2, min_samples_leaf=5,
    max_features="sqrt",
    mode="bootstrap", bag_fraction=0.7,
    random_state=42, oob_score=True,
    print_examples=True
):
    print("\n" + "="*70)
    print("Random Sherwood – settings")
    print(f"  n_estimators={n_estimators}, max_depth={max_depth}, "
          f"min_split={min_samples_split}, min_leaf={min_samples_leaf}, "
          f"max_features={max_features}, mode={mode}, bag_fraction={bag_fraction}, "
          f"oob_score={oob_score}")
    rf = RandomSherwoodForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        mode=mode,
        bag_fraction=bag_fraction,
        random_state=random_state,
        oob_score=oob_score
    )
    t0 = time.perf_counter()
    rf.fit(Xtr, Ytr)
    t1 = time.perf_counter()
    y_tr = rf.predict(Xtr)
    t2 = time.perf_counter()
    y_te = rf.predict(Xte)
    t3 = time.perf_counter()

    # metriker
    tr_acc = (y_tr == Ytr).mean()
    te_acc = (y_te == Yte).mean()
    fit_s = t1 - t0
    pred_tr_s = t2 - t1
    pred_te_s = t3 - t2
    print("\nResults")
    print(f"  OOB score: {rf.oob_score_:.1%}" if rf.oob_score_ is not None else "  OOB score: N/A")
    print(f"  Train acc: {tr_acc:.1%}")
    print(f"  Test  acc: {te_acc:.1%}")
    print(f"\nTime-efficiency")
    print(f"  Fit time:           {fit_s:.2f} s")
    print(f"  Predict train time: {pred_tr_s:.2f} s  ({len(Ytr)} samp → {(len(Ytr)/max(1e-9,pred_tr_s)):.1f} samp/s)")
    print(f"  Predict test  time: {pred_te_s:.2f} s  ({len(Yte)} samp → {(len(Yte)/max(1e-9,pred_te_s)):.1f} samp/s)")

    # confusion & exempel
    cm, labels = confusion_matrix(Yte, y_te)
    print()
    print_confusion_summary(cm, labels, top_k=10)
    plot_confusion_matrix(cm, labels, normalize=False)
    if print_examples:
        print()
        show_example_indices(Yte, y_te, n=4, kind="success")
        show_example_indices(Yte, y_te, n=4, kind="fail")

    show_fail_pairs_with_utils(
    Xte, Yte, y_te,
    Xtr, Ytr,
    utils_module=utils,
    n_pairs=4
    )
    visualize_success_fail(Xte, Yte, y_te, utils, n_show=4)
    #sweep_random_forest(Xtr, Ytr, Xte, Yte)
    return {
        "model": rf,
        "train_acc": tr_acc, "test_acc": te_acc, "oob": rf.oob_score_,
        "fit_time": fit_s, "pred_train_time": pred_tr_s, "pred_test_time": pred_te_s,
        "cm": cm, "labels": labels, "y_pred_test": y_te
    }

# --- 3) Liten param-svepare för “impact of parameters/weak-learners” ---
def sweep_random_forest(Xtr, Ytr, Xte, Yte,
                        n_estimators_list=(50),
                        max_depth_list=(12,18,22),
                        min_leaf_list=(3,5,10),
                        max_features_list=("sqrt"),
                        modes=("bootstrap","bagging")):
    print("\n" + "#"*70)
    print("# Parameter sweep (accuracy & tid) — kort rapport")
    print("#"*70)
    rows = []
    for ne in n_estimators_list:
        for md in max_depth_list:
            for ml in min_leaf_list:
                for mf in max_features_list:
                    for mode in modes:
                        res = run_experiment(
                            Xtr, Ytr, Xte, Yte,
                            n_estimators=ne,
                            max_depth=md,
                            min_samples_split=2,
                            min_samples_leaf=ml,
                            max_features=mf,
                            mode=mode,
                            bag_fraction=0.7,
                            random_state=42,
                            oob_score=True,
                            print_examples=False
                        )
                        rows.append((
                            ne, md, ml, str(mf), mode,
                            res["train_acc"], res["test_acc"], res["oob"],
                            res["fit_time"], res["pred_test_time"]
                        ))
    # kompakt tabell
    print("\nSummary table")
    print(" n_estim  depth  min_leaf  max_feat     mode     train   test    oob    fit_s  pred_te_s")
    for r in rows:
        ne, md, ml, mf, mode, tr, te, oob, ft, pt = r
        oob_str = "N/A" if oob is None else f"{oob:.3f}"
        print(f" {ne:7d}  {md:5d}    {ml:7d}  {mf:8s}  {mode:9s}  {tr:6.3f}  {te:6.3f}  {oob_str:>6s}  {ft:6.2f}  {pt:9.2f}")

# ====== LADDA DATA & KÖR ======
if __name__ == "__main__":
    x_data, y_labels = utils.getImages()
    Xtr, Xte, Ytr, Yte = utils.separateTrainingTestQ1(x_data, y_labels)

    Xtr = np.asarray(Xtr, dtype=np.float32, order="C")
    Xte = np.asarray(Xte, dtype=np.float32, order="C")
    Ytr = np.asarray(Ytr).ravel().astype(int)
    Yte = np.asarray(Yte).ravel().astype(int)
    if Xtr.shape[0] != len(Ytr) and Xtr.shape[1] == len(Ytr): Xtr = Xtr.T
    if Xte.shape[0] != len(Yte) and Xte.shape[1] == len(Yte): Xte = Xte.T

    # En “bra baseline”
    run_experiment(
    Xtr, Ytr, Xte, Yte,
    n_estimators=5, max_depth=18,
    min_samples_split=1, min_samples_leaf=5,
    max_features="sqrt",
    mode="bootstrap", bag_fraction=0.7,
    random_state=42, oob_score=True,
    print_examples=False
    )
    # Testing depth
   
    # (valfritt) svep för rapportens “impact of parameters”
    #sweep_random_forest(Xtr, Ytr, Xte, Yte)

    depth_time = np.array([10.9 , 34.40, 57.73, 63.5, 69.33, 66.58])
    depth_variables = np.array([2, 6, 10, 14, 18, 22])
    depth_result = np.array([14.4 , 40.4, 52.9, 69.85, 66.3,63.5])

    depth_time2 = [11.33 , 39.12, 69.83,  84.83, 89.71, 88.14]
    depth_variables2 = [2 ,6, 10, 14, 18, 22]
    depth_result2 = [14.4, 39.4, 61.5, 59.6, 62.5, 65.4]  
    # --- Linjär regression med np.polyfit ---
    # (y = a*x + b)
    koeff = np.polyfit(depth_variables, depth_time, 1)
    a, b = koeff
    print(f"Lutning (a): {a:.3f}")
    print(f"Intercept (b): {b:.3f}")

    # Skapa en linje för att plotta regressionen
    x_lin = np.linspace(min(depth_variables), max(depth_variables), 100)
    y_lin = a * x_lin + b

    # --- Plotta ---
    plt.figure(figsize=(8,6))
    plt.plot(depth_variables, depth_result, color='blue', label='Model accuracy, when min 5 leaves', marker='o', linestyle='-')
    plt.plot(depth_variables, depth_result2, color='red', label='Model accuracy, unregulated', marker='o', linestyle='-')
    #plt.scatter(depth_variables, depth_time, color='blue', label='Time measurements')
    #plt.plot(x_lin, y_lin, color='red', label='Linjär regression')
    plt.xlabel('Maximum tree depth allowed')
    
    plt.ylabel('Model accuracy in %')
    plt.title('Model accuracy vs Depth variables')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- R^2 (förklaringsgrad) ---
    y_pred = a * depth_variables + b
    ss_res = np.sum((depth_time - y_pred)**2)
    ss_tot = np.sum((depth_time - np.mean(depth_time))**2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"R^2 = {r2:.3f}")