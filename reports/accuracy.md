# Accuracy on the samples

- **dots found**: ground-truth dots with a detected dot within 1.5 dot radii
- **numbers read**: numbers read correctly *next to their own dot*
- **wrong readings**: OCR numbers with no matching dot (misreads, duplicates)
- **numbers placed**: numbers that end up on the correct dot after matching (what the animation draws)

## OCR backend: paddle (31 s/image)

| sample                 | dots found     | numbers read    | wrong readings | numbers placed  |
|------------------------|----------------|-----------------|----------------|-----------------|
| commons_crow           | 58/58 (100%)   | 58/58 (100%)    | 0/58           | 58/58 (100%)    |
| commons_dragon         | 100/100 (100%) | 100/100 (100%)  | 0/100          | 100/100 (100%)  |
| commons_hydra          | 173/173 (100%) | 173/173 (100%)  | 0/173          | 173/173 (100%)  |
| commons_kiwi_bird      | 46/46 (100%)   | 46/46 (100%)    | 0/46           | 46/46 (100%)    |
| commons_seahorse       | 58/58 (100%)   | 58/58 (100%)    | 0/58           | 58/58 (100%)    |
| commons_secretary_bird | 72/72 (100%)   | 72/72 (100%)    | 1/74           | 72/72 (100%)    |
| commons_trumpet        | 158/158 (100%) | 158/158 (100%)  | 0/158          | 158/158 (100%)  |
| picture_1              | 50/50 (100%)   | 50/50 (100%)    | 0/51           | 50/50 (100%)    |
| picture_2              | 47/47 (100%)   | 47/47 (100%)    | 2/49           | 47/47 (100%)    |
| web_1                  | 50/50 (100%)   | 43/50 (86.0%)   | 8/52           | 39/50 (78.0%)   |
| TOTAL                  | 812/812 (100%) | 805/812 (99.1%) | 11/819         | 801/812 (98.6%) |
