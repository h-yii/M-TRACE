from collections import defaultdict

VALID_OPTIONS = {"A", "B", "C", "D", "E"}


class QAStats:
    def __init__(self):

        self.type_total = defaultdict(int)
        self.type_correct = defaultdict(int)
        self.type_wrong = defaultdict(int)
        self.type_other = defaultdict(int)

        self.disturb_total = defaultdict(int)
        self.disturb_correct = defaultdict(int)
        self.disturb_wrong = defaultdict(int)
        self.disturb_other = defaultdict(int)

        self.global_total = 0
        self.global_correct = 0
        self.global_wrong = 0
        self.global_other = 0

    def update_stats(self, item, pred):

        correct = item["correct_option"]
        wrong = item["wrong_option"]

        qtype = item["type"]
        disturb = item["disturb_class"]

        is_valid = pred in VALID_OPTIONS
        is_correct = is_valid and pred == correct
        is_wrong = is_valid and pred == wrong
        is_other = is_valid and (pred != correct) and (pred != wrong)

        self.global_total += 1
        if is_correct:
            self.global_correct += 1
        elif is_wrong:
            self.global_wrong += 1
        elif is_other:
            self.global_other += 1

        self.type_total[qtype] += 1
        if is_correct:
            self.type_correct[qtype] += 1
        elif is_wrong:
            self.type_wrong[qtype] += 1
        elif is_other:
            self.type_other[qtype] += 1

        self.disturb_total[disturb] += 1
        if is_correct:
            self.disturb_correct[disturb] += 1
        elif is_wrong:
            self.disturb_wrong[disturb] += 1
        elif is_other:
            self.disturb_other[disturb] += 1

    def report(self):
        print("\n==============================")
        print("        SUMMARY REPORT")
        print("==============================\n")

        # =====================================================
        # 1. 按 TYPE
        # =====================================================
        for t in sorted(self.type_total.keys()):
            total = self.type_total[t]
            c = self.type_correct[t]
            w = self.type_wrong[t]
            o = self.type_other[t]

            print(f"- Type: {t}")
            print(f"    Correct: {c}/{total} ({c/total:.2%})")
            print(f"    Wrong:   {w}/{total} ({w/total:.2%})")
            print(f"    Other:   {o}/{total} ({o/total:.2%})")

        # =====================================================
        # 2. 按 Disturb Class
        # =====================================================
        for d in sorted(self.disturb_total.keys()):
            total = self.disturb_total[d]
            c = self.disturb_correct[d]
            w = self.disturb_wrong[d]
            o = self.disturb_other[d]

            print(f"- Disturb: {d}")
            print(f"    Correct: {c}/{total} ({c/total:.2%})")
            print(f"    Wrong:   {w}/{total} ({w/total:.2%})")
            print(f"    Other:   {o}/{total} ({o/total:.2%})")

        # =====================================================
        # 3. 全局
        # =====================================================
        g = self.global_total
        print(f"    Correct: {self.global_correct}/{g} ({self.global_correct/g:.2%})")
        print(f"    Wrong:   {self.global_wrong}/{g} ({self.global_wrong/g:.2%})")
        print(f"    Other:   {self.global_other}/{g} ({self.global_other/g:.2%})")

        print("\n==============================\n")
