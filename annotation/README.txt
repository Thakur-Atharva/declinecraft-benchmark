DeclineCraft annotation tool (human validation)
===============================================

This is the tool each annotator received. It needs Python 3.8 or later and nothing else.

1. Put a labels_<name>.csv item file next to annotate.py (one per annotator; not included here).
2. Run:   python annotate.py        (or: python3 annotate.py)
3. The browser opens the annotation page. Read the Guidelines tab first.
4. Every answer is saved straight into labels_<name>.csv. Stop with Ctrl+C; run step 2 again to carry on.
5. When both parts are finished, the annotator sends back labels_<name>.csv.

Item file columns:
  item_id, annotator, task, question, answer, reference, b_response, b_correct, note, updated_at
  task        R = part 1 (label the response), C = part 2 (score accuracy against the reference)
  b_response  filled by the annotator in part 1: refusal | premise | attempt
              (= ABSTAIN | DENY | ANSWER of the response judge)
  b_correct   filled by the annotator in part 2: 1 | 0.5 | 0
  reference   the reference answer (part 2 only)
The page shows only the question, the answer and, for part 2, the reference; never the model or any judge label.
