from counterfact.suites import answer_change_stats, generate_suite

if __name__ == "__main__":
    suite = generate_suite()
    print(f"Wrote data/suites/hard_v1.json ({len(suite.families)} synthetic families)")
    print(answer_change_stats(suite))
