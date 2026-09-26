# Shadowboxing AI

A personal shadowboxing coach for an M3 MacBook Pro and its built-in webcam, with local live analysis and optional cloud review of selected clips.

**Status: research and implementation plan, September 26, 2026.** There is no working camera app or validated coaching model yet. No camera footage has been captured, and no model performance has been measured on this machine.

The recommended approach combines body pose tracking, temporal punch recognition, explicit coaching criteria, and evidence-linked feedback. Technique recognition and technique assessment are evaluated separately. Unobservable or uncertain movements produce an abstention, never a guessed correction.

Start with the [project plan](docs/project-plan.md), then the [first benchmark sprint](docs/first-sprint.md).

Research notes:

- [Pose models and deployment](docs/research/pose-models.md)
- [Boxing data, action recognition, and evaluation](docs/research/data-and-evaluation.md)
- [Coaching validity and camera limitations](docs/research/coaching-validity.md)
- [Video-language models and review architecture](docs/research/video-review.md)

The first deliverable should be a camera and replay lab that establishes what can actually be measured. Guided jab/cross coaching follows only after the recognition and feedback gates pass.

Raw recordings, personal datasets, credentials, and model weights stay out of Git. Repository documentation is not a license grant for third-party datasets or weights.
