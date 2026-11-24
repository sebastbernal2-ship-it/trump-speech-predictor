# Distilled annotator

## Pipeline

The local pipeline has four stages:

1. The data builder turns frozen annotation drafts into training examples.
2. The LoRA trainer fine-tunes Qwen3-0.6B on those examples.
3. The embedding matcher compares candidate annotations and uses separately calibrated thresholds for each layer.
4. Graph evaluation checks annotation structure, including key-point attachment to parent ideas.

## Five-fold results on 67 frozen drafts

| Layer or check | Result |
| --- | --- |
| Ideas | Recall 72%, precision 79% at threshold 0.50 |
| Key points | Recall 39%, precision 60% |
| Topics | Recall 41%, precision 45% |
| Opinions | Recall 47%, precision 55% |
| Tone | Recall 50%, precision 59% |
| Quotes | 90.8% exact |
| Views with broken links | 22% |
| Key points attached to the correct parent | 57% |

## Limits

- The AI-judge labels have not been user-reviewed.
- Evaluation uses only 67 labeled drafts.
- The cosine matcher has strict F1 of about 0.75.
- Qwen3-4B and the NLI matcher were not run.
- Transcript annotations are outcomes for their own event and must never be same-event prediction inputs.
