# Characters stage (reserved)

**Status:** scaffold only; no character-level model or task is implemented.

This stage concerns literal text characters, not people or fictional characters. Before implementation, define the unit (Unicode code point or grapheme cluster), normalization, encoding, and treatment of whitespace and punctuation. Bind every artifact to that contract/version.

Character-level likelihood or generation metrics are not substitutes for word, sentence, or idea quality. If conditioned on a word artifact, consume its frozen serialized text; report teacher-forced and end-to-end results separately.