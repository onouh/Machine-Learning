"""ADR regression experiments and inference pipeline.

Target: ADR (average daily rate) at booking creation.
This package houses all regression-model work: data cleaning, preprocessing,
feature selection, clustering, model training, and evaluation — all gated
through the booking-time-only 'after' variant to avoid information leakage.
"""
