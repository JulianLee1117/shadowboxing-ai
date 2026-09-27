# Next short capture for broader recognition

The current v7 preview can emit jabs, crosses, lead/rear hooks and lead/rear uppercuts using experimental projected-motion rules. Hooks and uppercuts can still be missed entirely when tracking is obscured or the visible path differs from those rules. These optional captures supply development examples and hard negatives; they do not establish six-punch accuracy or correct form.

Use **Free practice**, the correct **Lead hand**, and the existing 30-second round. Keep a comfortable, slightly angled view with chest and both arms visible. Use **Focus view** if useful and the automatic countdown to step back. Keep one stance for each recording. The live punch name and count are predictions; do not change the reference labels to agree with them.

1. **Hooks:** a few comfortable lead hooks, then a few rear hooks, with a brief pause between actions. Spend the final 8 seconds resting, changing guard and moving naturally without punching.
2. **Uppercuts:** a few comfortable lead uppercuts, then a few rear uppercuts, with pauses. Again leave the final 8 seconds for ordinary non-punch movement.

No exaggerated speed or intentionally poor technique is needed. Retain misses, imperfect attempts and incidental movements in the original recording. Intended punch names and requested counts are not reference labels: label the visible action and anatomical hand from video, mark genuinely unobservable intervals, and confirm completeness only after reviewing the whole clip. A visible imperfect hook is still an action example, not an automatic technique failure.

Keep video plus **Evidence JSON** locally. In the larger Review player, use **Focus video**, slow playback and frame stepping to label the video first, with **Show detections** off. Enable detections afterward to compare the six count cards, timeline and punch list. A zero count does not prove no punch occurred. These first examples become development data. After a recognizer and protocol are frozen, reserve a later-day round for evaluation before inspecting its predictions or tuning on it. More people, stances, views and environments are needed before claiming the app works broadly.

The [dataset preparation tool](action-dataset.md) tracks class gaps and prevents incomplete annotations from silently becoming background. Correct technique needs the separate [coach-review workflow](coaching-review.md); copying a person's habits is not sufficient supervision for coaching.
