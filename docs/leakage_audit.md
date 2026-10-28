# Leakage audit

Status: preliminary; update after feature availability and prediction timing are confirmed. This is not evidence that a model has been trained.

## Regression: ADR

- Target: `ADR`.
- Confirm the prediction point (for example, booking creation) before choosing predictors. A feature recorded after rate assignment or after booking changes may not be available then.
- Never include `ADR` itself, or transformations derived from it, among predictors.
- `ReservationStatus` and `ReservationStatusDate` describe a later reservation outcome/status process and should not be used for a booking-time ADR estimate without a justified use case.
- Compare analyses with ADR=0 included and excluded; the business meaning of these 1,208 records is unresolved.

## Classification: IsCanceled

- Assumed prediction point: booking creation; verify this against the PMS workflow.
- `ReservationStatus` is direct outcome leakage: in the supplied CSV it agrees exactly with `IsCanceled` (`Check-Out` corresponds to 0; `Canceled` and `No-Show` correspond to 1).
- Exclude `ReservationStatusDate` from booking-creation prediction.
- Exclude fields that can accrue or change after creation from the creation-time baseline, including `BookingChanges`, `DaysInWaitingList`, and `AssignedRoomType`, unless source timing proves they are already available at the defined prediction point.
- Check availability timing for `ADR`, `RequiredCarParkingSpaces`, `TotalOfSpecialRequests`, and other operational fields before including them.

## Clustering

- Keep the process unsupervised: do not include `IsCanceled`, `ReservationStatus`, or `ReservationStatusDate` in the feature representation used to discover booking segments.
- Exclude future/post-booking outcomes and document any timing assumptions for booking attributes.
- The data has no customer key; interpret clusters as booking profiles, not longitudinal customer segments.

## Workflow controls still to implement

- Fit imputers, encoders, scalers, feature selection, PCA/LDA, and other learned transforms on training data only for supervised tasks.
- Keep tuning inside training/CV; do not use the final test set for model selection.
- Do not use supervised target labels to tune clustering parameters.
- Record final feature allowlists, excluded fields, and justification per task before evaluating models.
