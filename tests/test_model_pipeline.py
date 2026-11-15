import unittest

import pandas as pd

from model_pipeline import (
    FEATURE_SETS,
    add_booking_features,
    build_preprocessor,
    validate_splits,
)


class BookingFeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "scheduled_day": pd.to_datetime(
                    ["2016-06-03 14:00", "2016-06-01 09:00"]
                ),
                "appointment_day": pd.to_datetime(
                    ["2016-06-03", "2016-07-02"]
                ),
                "age": [21, 115],
                "neighbourhood": ["Clinic A", "Clinic B"],
                "prev_appointment_count": [2, 0],
                "prev_no_show_rate": [0.5, 0.0],
            }
        )

    def test_derives_booking_time_features(self):
        result = add_booking_features(self.frame)

        self.assertEqual(result["lead_time_days"].tolist(), [0, 31])
        self.assertEqual(result["lead_bucket"].tolist(), ["Same-day (0d)", "31+ days"])
        self.assertEqual(result["age_group"].tolist(), ["13-24", "66+"])
        self.assertEqual(result["same_day_has_history"].tolist(), [1, 0])
        self.assertEqual(result["same_day_prior_no_show_rate"].tolist(), [0.5, 0.0])
        self.assertEqual(
            result["same_day_age_group"].tolist(), ["13-24", "Not same-day"]
        )
        self.assertEqual(
            result["same_day_neighbourhood"].tolist(),
            ["Clinic A", "Not same-day"],
        )
        self.assertAlmostEqual(result["lead_time_log1p"].iloc[1], 3.4657359)

    def test_preprocessor_ignores_unseen_categories(self):
        result = add_booking_features(self.frame)
        features = list(FEATURE_SETS["booking_context"])
        preprocessor = build_preprocessor(FEATURE_SETS["booking_context"])
        train_matrix = preprocessor.fit_transform(result.iloc[:1][features])
        test_matrix = preprocessor.transform(result.iloc[1:][features])

        self.assertEqual(train_matrix.shape[1], test_matrix.shape[1])
        self.assertEqual(test_matrix.shape[0], 1)


class SplitValidationTests(unittest.TestCase):
    @staticmethod
    def split(appointment_id, appointment_day):
        return pd.DataFrame(
            {
                "appointment_id": [appointment_id],
                "appointment_day": [pd.Timestamp(appointment_day)],
                "no_show": [0],
            }
        )

    def test_accepts_disjoint_chronological_splits(self):
        validate_splits(
            self.split(1, "2016-05-01"),
            self.split(2, "2016-05-02"),
            self.split(3, "2016-05-03"),
        )

    def test_accepts_selection_splits_without_loading_test(self):
        validate_splits(
            self.split(1, "2016-05-01"),
            self.split(2, "2016-05-02"),
        )

    def test_rejects_test_appointment_id_overlap(self):
        with self.assertRaisesRegex(ValueError, "appointment IDs overlap"):
            validate_splits(
                self.split(1, "2016-05-01"),
                self.split(2, "2016-05-02"),
                self.split(2, "2016-05-03"),
            )

    def test_rejects_nonchronological_test_split(self):
        with self.assertRaisesRegex(ValueError, "not strictly chronological"):
            validate_splits(
                self.split(1, "2016-05-01"),
                self.split(2, "2016-05-02"),
                self.split(3, "2016-05-02"),
            )

    def test_rejects_duplicate_appointment_ids(self):
        train = pd.concat(
            [
                self.split(1, "2016-05-01"),
                self.split(1, "2016-05-01"),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate appointment IDs"):
            validate_splits(
                train,
                self.split(2, "2016-05-02"),
                self.split(3, "2016-05-03"),
            )


if __name__ == "__main__":
    unittest.main()
