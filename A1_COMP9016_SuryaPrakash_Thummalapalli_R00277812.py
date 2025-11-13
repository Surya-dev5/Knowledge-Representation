from logic import FolKB, expr, fol_fc_ask, fol_bc_ask
from probability import BayesNet, enumeration_ask, elimination_ask, T, F

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
            if not list(fol_bc_ask(fol_kb, expr(f"Passed({s}, {m})"))):
                fol_kb.tell(expr(f"NotPassed({s}, {m})"))


    # RULES

    fol_kb.tell(expr("(Takes(s,m) & Teaches(l,m)) ==> TaughtBy(s,l)"))
    fol_kb.tell(expr("(Takes(s,m) & Takes(t,m) & NotSame(s,t)) ==> Classmate(s,t)"))
    fol_kb.tell(expr("(Prereq(p,m) & Passed(s,p)) ==> Fulfilled(s,p,m)"))
    fol_kb.tell(expr("(AllPrereqsPassed(s,m)) ==> Eligible(s,m)"))

    prereqs = {("COMP9016", "COMP9062"), ("COMP9062", "COMP9061"), ("COMP9016", "COMP9061")}
    for s in students:
        for m in modules:
            ps = [p for (p,mm) in prereqs if mm == m]
            if ps and all(list(fol_bc_ask(fol_kb, expr(f"Passed({s},{p})"))) for p in ps):
                fol_kb.tell(expr(f"AllPrereqsPassed({s},{m})"))
                fol_kb.tell(expr(f"Eligible({s},{m})"))

    fol_kb.tell(expr("(Eligible(s,m) & NotPassed(s,m)) ==> CanEnroll(s,m)"))

    fol_kb.tell(expr("(Prereq(p,m) & Prereq(m,n)) ==> IndirectPrereq(p,n)"))

    # FORWARD CHAINING
    print("\n=== Forward Chaining Derived Facts ===")
    for q in ["TaughtBy(x,y)", "Classmate(x,y)", "Eligible(x,y)", "CanEnroll(x,y)", "IndirectPrereq(x,y)"]:
        results = list(fol_fc_ask(fol_kb, expr(q)))
        if results:
            print(q, "=>", results)


    # BACKWARD CHAINING
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


    bn = BayesNet([
        ('HypeLevel', '', 0.6),
        ('VCInvestment', 'HypeLevel', {T: 0.75, F: 0.20}),
        ('ComputeCosts', 'HypeLevel', {T: 0.65, F: 0.30}), 
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

q1_1_logical_reasoning()
q1_2_bayesian_network()