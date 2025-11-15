from logic import FolKB, expr, fol_fc_ask, fol_bc_ask
from probability import BayesNet, enumeration_ask, elimination_ask, T, F
import numpy as np
import math
from collections import Counter
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def q1_1_logical_reasoning():

    fol_kb = FolKB()

# FACTS
    modules   = ["COMP9016", "COMP9062", "COMP9061", "COMP9058"]
    students  = ["Alice", "Bob", "Eve"]
    lecturers = ["DrDan", "DrSophie", "DrLisa"]

    for m in modules:
        fol_kb.tell(expr(f"Module({m})"))
    for s in students:
        fol_kb.tell(expr(f"Student({s})"))
    for l in lecturers:
        fol_kb.tell(expr(f"Lecturer({l})"))

    # prerequisites
    fol_kb.tell(expr("Prereq(COMP9016, COMP9062)"))
    fol_kb.tell(expr("Prereq(COMP9062, COMP9061)"))
    # transitive prerequisite to avoid recursion
    fol_kb.tell(expr("Prereq(COMP9016, COMP9061)"))

    # teaching
    fol_kb.tell(expr("Teaches(DrDan, COMP9016)"))
    fol_kb.tell(expr("Teaches(DrSophie, COMP9062)"))
    fol_kb.tell(expr("Teaches(DrLisa, COMP9061)"))

    # student record
    fol_kb.tell(expr("Passed(Alice, COMP9016)"))
    fol_kb.tell(expr("Passed(Alice, COMP9058)"))
    fol_kb.tell(expr("Passed(Bob, COMP9016)"))
    fol_kb.tell(expr("Passed(Bob, COMP9062)"))

    # current enrolments
    fol_kb.tell(expr("Takes(Alice, COMP9062)"))
    fol_kb.tell(expr("Takes(Bob, COMP9061)"))
    fol_kb.tell(expr("Takes(Eve, COMP9016)"))

    # explicit NotSame facts (Horn clause limitation)
    fol_kb.tell(expr("NotSame(Alice, Bob)"))
    fol_kb.tell(expr("NotSame(Alice, Eve)"))
    fol_kb.tell(expr("NotSame(Bob, Eve)"))
    fol_kb.tell(expr("NotSame(Bob, Alice)"))
    fol_kb.tell(expr("NotSame(Eve, Alice)"))
    fol_kb.tell(expr("NotSame(Eve, Bob)"))

    # explicit NotPassed facts (negation workaround)
    for s in students:
        for m in modules:
            # mark as not passed if not already known as Passed
            if not list(fol_bc_ask(fol_kb, expr(f"Passed({s}, {m})"))):
                fol_kb.tell(expr(f"NotPassed({s}, {m})"))

    # -----------------------------------------------------------
    # RULES (Horn clauses only)
    # -----------------------------------------------------------

    # 1) Teaching implies "taught by" when student enrolled
    fol_kb.tell(expr("(Takes(s,m) & Teaches(l,m)) ==> TaughtBy(s,l)"))

    # 2) Classmates if share module and not same person
    fol_kb.tell(expr("(Takes(s,m) & Takes(t,m) & NotSame(s,t)) ==> Classmate(s,t)"))

    # 3) Eligible if student has passed all prerequisites of module
    fol_kb.tell(expr("(Prereq(p,m) & Passed(s,p)) ==> Fulfilled(s,p,m)"))
    fol_kb.tell(expr("(AllPrereqsPassed(s,m)) ==> Eligible(s,m)"))

    # pre-compute AllPrereqsPassed manually for this finite set
    prereqs = {("COMP9016", "COMP9062"), ("COMP9062", "COMP9061"), ("COMP9016", "COMP9061")}
    for s in students:
        for m in modules:
            ps = [p for (p,mm) in prereqs if mm == m]
            if ps and all(list(fol_bc_ask(fol_kb, expr(f"Passed({s},{p})"))) for p in ps):
                fol_kb.tell(expr(f"AllPrereqsPassed({s},{m})"))
                fol_kb.tell(expr(f"Eligible({s},{m})"))

    # 4) CanEnroll if eligible and not passed
    fol_kb.tell(expr("(Eligible(s,m) & NotPassed(s,m)) ==> CanEnroll(s,m)"))

    # 5) Indirect prerequisite (non-recursive version)
    fol_kb.tell(expr("(Prereq(p,m) & Prereq(m,n)) ==> IndirectPrereq(p,n)"))

    # -----------------------------------------------------------
    # FORWARD CHAINING
    # -----------------------------------------------------------
    print("\n=== Forward Chaining Derived Facts ===")
    for q in ["TaughtBy(x,y)", "Classmate(x,y)", "Eligible(x,y)", "CanEnroll(x,y)", "IndirectPrereq(x,y)"]:
        results = list(fol_fc_ask(fol_kb, expr(q)))
        if results:
            print(q, "=>", results)

    # -----------------------------------------------------------
    # BACKWARD CHAINING
    # -----------------------------------------------------------
    print("\n=== Backward Chaining Queries ===")
    queries = [
        "Eligible(Alice, COMP9061)",
        "Eligible(Bob, COMP9061)",
        "TaughtBy(Eve, l)",
        "Classmate(Alice, Eve)"
    ]
    for q in queries:
        ans = list(fol_bc_ask(fol_kb, expr(q)))
        print(f"{q} =>", ans if ans else "No proof found")

    fol_kb.tell(expr("Takes(Alice, COMP9016)"))
    print("\nAfter adding Takes(Alice, COMP9016):")
    ans2 = list(fol_bc_ask(fol_kb, expr("Classmate(Alice, Eve)")))
    print("Classmate(Alice, Eve) =>", ans2 if ans2 else "No proof found")


    return fol_kb
def q1_2_bayesian_network():
    """
    Constructs a small Bayesian Network for the AI-market scenario and
    performs example queries demonstrating d-separation.
    """

    # DAG structure:
    # HypeLevel → {VCInvestment, ComputeCosts, EnterpriseAdoption}
    # EnterpriseAdoption → {RegulatoryPressure, LabourMarketImpact}
    # RegulatoryPressure → LabourMarketImpact
    bn = BayesNet([
        ('HypeLevel', '', 0.6),
        ('VCInvestment', 'HypeLevel', {T: 0.75, F: 0.20}),
        ('ComputeCosts', 'HypeLevel', {T: 0.65, F: 0.30}),  # True = low costs
        ('EnterpriseAdoption', 'HypeLevel VCInvestment ComputeCosts', {
            (T,T,T): 0.90, (T,T,F): 0.70, (T,F,T): 0.70, (T,F,F): 0.40,
            (F,T,T): 0.60, (F,T,F): 0.30, (F,F,T): 0.25, (F,F,F): 0.10}),
        ('RegulatoryPressure', 'HypeLevel EnterpriseAdoption', {
            (T,T): 0.80, (T,F): 0.50, (F,T): 0.60, (F,F): 0.20}),
        ('LabourMarketImpact', 'EnterpriseAdoption RegulatoryPressure', {
            (T,T): 0.85, (T,F): 0.65, (F,T): 0.40, (F,F): 0.10})
    ])

    # Example queries
    q1 = enumeration_ask('EnterpriseAdoption', dict(HypeLevel=T), bn)
    q2 = enumeration_ask('LabourMarketImpact', dict(HypeLevel=T, VCInvestment=T), bn)
    q3 = elimination_ask('HypeLevel', dict(RegulatoryPressure=T), bn)

    print("\n=== Part 1.2 Bayesian Network Results ===")
    print("P(EnterpriseAdoption | HypeLevel=T):", q1.prob)
    print("P(LabourMarketImpact | HypeLevel=T, VCInvestment=T):", q2.prob)
    print("P(HypeLevel | RegulatoryPressure=T):", q3.prob)


    return bn

class GaussianNaiveBayes:
    """
    Simple Gaussian Naïve Bayes classifier:
    - Uses mean/variance per class (continuous features)
    - Assumes conditional independence of features given class
    """
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means_ = {}
        self.vars_ = {}
        self.priors_ = {}

        for c in self.classes_:
            Xc = X[y == c]
            self.means_[c] = Xc.mean(axis=0)
            self.vars_[c] = Xc.var(axis=0) + self.var_smoothing
            self.priors_[c] = len(Xc) / len(X)
        return self
    
    def gaussian_log_likelihood(self, x, c):
        mean = self.means_[c]
        var = self.vars_[c]
        return -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)

    def predict(self, X):
        predictions = []
        for x in X:
            log_probs = {}
            for c in self.classes_:
                # log P(c) + log P(x | c)
                log_probs[c] = math.log(self.priors_[c]) + self.gaussian_log_likelihood(x, c)
            predictions.append(max(log_probs, key=log_probs.get))
        return np.array(predictions)

def q1_3_naive_bayes_classification():
    """
    PART 1.3 IMPLEMENTATION AND ANALYSIS OF NAIVE BAYES CLASSIFIERS

    1.3.1 DATA SELECTION AND PREPROCESSING
        - Select two UCI datasets (Iris, Wine).
        - Compute class priors P(c).
        - For an example x0, compute:
            * likelihood P(x0 | c) for each class
            * numerator P(x0 | c) P(c)
            * evidence P(x0) = sum_c P(x0 | c) P(c)
        - Explain how these relate to Bayes' theorem and Naive Bayes.

    1.3.2 NAIVE BAYES CLASSIFICATION
        - Implement Gaussian Naive Bayes.
        - Train and evaluate on both datasets.
        - Print accuracy and basic diagnostics
          (visualizations can be done easily from these results in a notebook).
    """

    try:
        from sklearn import datasets
    except ImportError:
        print("scikit-learn not available: cannot load UCI datasets (Iris, Wine).")
        print("Please install scikit-learn or run in the provided environment.")
        return

    # ---------- DATA SELECTION: Two UCI datasets (via sklearn) ----------
    iris = datasets.load_iris()   # 150 samples, 4 features, 3 classes
    wine = datasets.load_wine()   # 178 samples, 13 features, 3 classes

    datasets_to_use = [
        ("Iris", iris.data, iris.target),
        ("Wine", wine.data, wine.target)
    ]

    print("\n=== PART 1.3 – Naive Bayes: Data Analysis and Classification ===")

    for name, X, y in datasets_to_use:
        print(f"\n--- Dataset: {name} ---")
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # 1) Calculate prior probabilities P(c) for the full dataset
        counts = Counter(y)
        priors = {c: counts[c] / n_samples for c in counts}
        print("Number of samples:", n_samples)
        print("Number of features:", n_features)
        print("Class counts:", dict(counts))
        print("Class priors P(c):", priors)

        # 2) Train/test split (simple hold-out: 70% train, 30% test)
        idx = np.arange(n_samples)
        np.random.shuffle(idx)
        split = int(0.7 * n_samples)
        train_idx, test_idx = idx[:split], idx[split:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 3) Train Gaussian Naive Bayes
        gnb = GaussianNaiveBayes()
        gnb.fit(X_train, y_train)

        # 4) Evaluate classification performance
        y_pred = gnb.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print("Test accuracy (Gaussian NB):", round(accuracy, 4))

        # 5) Pick one evidence vector x0 from test set and analyze Bayes terms
        x0 = X_test[0]
        y0_true = y_test[0]
        print("Example evidence x0 (first test sample):", x0)
        print("True class of x0:", y0_true)

        likelihoods = {}
        numerators = {}
        for c in gnb.classes_:
            # log P(x0 | c) and P(x0 | c)
            log_lik = gnb.gaussian_log_likelihood(x0, c)
            lik = math.exp(log_lik)                   # P(x0 | c)
            likelihoods[c] = lik

            # numerator P(x0 | c) P(c)
            num = lik * gnb.priors_[c]
            numerators[c] = num

        # Evidence P(x0) = sum_c P(x0 | c) P(c)
        evidence = sum(numerators.values())

        print("Likelihoods P(x0 | c):", likelihoods)
        print("Numerators P(x0 | c) P(c):", numerators)
        print("Evidence P(x0):", evidence)

        # Posterior distribution for x0 (for interpretation)
        posteriors = {c: numerators[c] / evidence for c in gnb.classes_}
        print("Posterior P(c | x0):", posteriors)
        
q1_1_logical_reasoning()
q1_2_bayesian_network()
q1_3_naive_bayes_classification()