"""Independent prediction/generation stages by speech-content granularity.

The current hierarchy is topics -> ideas -> sentences -> words -> characters.
Only the ideas stage is implemented; each stage owns a separate contract and
versioned artifact boundary.
"""
