# Next short capture for broader recognition

The live detector currently recognizes jabs and crosses only. Hooks and uppercuts can already be labeled in Review, but their absence from the displayed count is expected until a broader model is trained and evaluated. These recordings supply examples; they are not an accuracy test of a hook/uppercut detector or a form assessment.

Use **Free practice**, the correct **Lead hand**, and the existing 30-second round. Keep a comfortable, slightly angled view with chest and both arms visible. Use the automatic countdown to step back. Keep one stance for each recording.

1. **Hooks:** a few comfortable lead hooks, then a few rear hooks, with a brief pause between actions. Spend the final 8 seconds resting, changing guard and moving naturally without punching.
2. **Uppercuts:** a few comfortable lead uppercuts, then a few rear uppercuts, with pauses. Again leave the final 8 seconds for ordinary non-punch movement.

No exaggerated speed or intentionally poor technique is needed. Retain misses, imperfect attempts and incidental movements in the original recording. Intended punch names and requested counts are not reference labels: label the visible action and anatomical hand from video, mark genuinely unobservable intervals, and confirm completeness only after reviewing the whole clip. A visible imperfect hook is still an action example, not an automatic technique failure.

Keep video plus Evidence JSON locally. These first examples become development data. After a model and protocol are frozen, reserve a later-day round for evaluation before inspecting its predictions or tuning on it. More people, stances, views and environments are needed before claiming the trainer works broadly.

The [dataset preparation tool](action-dataset.md) tracks class gaps and prevents incomplete annotations from silently becoming background. Correct technique needs the separate [coach-review workflow](coaching-review.md); copying a person's habits is not sufficient supervision for coaching.
