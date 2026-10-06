# Sample data

`06_10_26.mp4` is a 6-second phone video of "La Pista Cifrata" (56 dots) from
*La Settimana Enigmistica*, filmed on 6 October 2026. The pipeline solves it and
the AI says "a deer drinking from a tree":

```bash
uv run python run.py data/06_10_26.mp4 --start 4
```

More test puzzles, with exact ground truth for the accuracy report, come from
Wikimedia Commons (CC0 / CC BY-SA, see the `SOURCES.md` the script writes):

```bash
uv run python tools/fetch_samples.py
uv run python evaluate.py
```
