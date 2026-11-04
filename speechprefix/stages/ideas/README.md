# Ideas stage (scaffold)

**Status:** reserved package boundary; the active implementation has not yet been migrated here. It currently lives in `speechprefix/forecasting/context_to_ideas.py`.

This stage owns prediction of target-speech ideas from preceding-week context. Its future public output should be a versioned idea-forecast contract, with training and evaluation kept inside this stage. Do not add sentence realization or speech-history recurrence features here. Migration should preserve the current CLI/API through explicit compatibility imports and tests.
