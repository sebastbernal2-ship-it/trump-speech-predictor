# GovInfo sample audit

The local ignored sample contains 51 unique CPD records. It spans Jan 2017 to Sep 2026, with an uneven, title-filtered sample: 25 records from Sep 2026, one from May 2025, and 25 records from Jan 2017 through Sep 2020. It is not a representative or complete corpus.

A title-only classifier reports 5 `formal_address`, 15 `event_remarks`, 21 `remarks_plus_press`, 4 `press_interaction`, 6 `written_statement`, and 0 `other` records.

- `formal_address`: titles containing "address" or the United Nations General Assembly.
- `event_remarks`: titles mentioning remarks, without a reporter-exchange marker.
- `remarks_plus_press`: remarks titles that also mention an exchange with reporters.
- `press_interaction`: news conferences and reporter-focused titles.
- `written_statement`: titles beginning with "Statement" or "Joint Statement".
- `other`: titles that do not match the above.

Run `python3 -m speechprefix audit --input data/speeches.jsonl` to review each title and category. These labels are heuristics only. A long reporter exchange is not necessarily a prepared speech, and a short title may still correspond to substantial remarks. The full documents must be reviewed before calling a record a complete speech or including it in a clean speech benchmark.

The source sample is imbalanced by time because collection was bounded in separate runs. Do not use it for meaningful train/test claims. The current n-gram test score is a smoke test on the last chronologically sorted document, not a credible estimate of model performance.
