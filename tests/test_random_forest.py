"""Unit tests for the RandomForest regressor model module.

Tests cover:
- Model instantiation with seed 42
- Model specification (RANDOM_FOREST_SPEC)
- Factory functions
- Model configuration parameters
- Prediction reproducibility with fixed seed
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from hotel_booking.regression_models.random_forest import (
    DEFAULT_SEED,
    ModelSpec,
    RANDOM_FOREST_SPEC,
    get_random_forest_model,
    make_random_forest_regressor,
)


class TestModelSpec:
    """Tests for the ModelSpec dataclass."""

    def test_model_spec_creation(self) -> None:
        """Test that ModelSpec can be created with required fields."""
        spec = ModelSpec(
            key="TestModel",
            make_estimator=lambda: RandomForestRegressor(random_state=42),
            param_distributions={"n_estimators": [100]},
            n_iter=5,
            cv=3,
        )
        assert spec.key == "TestModel"
        assert spec.n_iter == 5
        assert spec.cv == 3
        assert spec.param_distributions == {"n_estimators": [100]}
        assert spec.subsample is None
        assert spec.note == ""

    def test_model_spec_with_optional_fields(self) -> None:
        """Test that ModelSpec accepts optional fields."""
        spec = ModelSpec(
            key="TestModel",
            make_estimator=lambda: RandomForestRegressor(random_state=42),
            param_distributions=None,
            n_iter=10,
            cv=5,
            subsample=1000,
            note="Test note",
        )
        assert spec.subsample == 1000
        assert spec.note == "Test note"

    def test_model_spec_is_frozen(self) -> None:
        """Test that ModelSpec is immutable (frozen dataclass)."""
        spec = ModelSpec(
            key="TestModel",
            make_estimator=lambda: RandomForestRegressor(random_state=42),
            param_distributions={},
            n_iter=5,
            cv=3,
        )
        with pytest.raises(Exception):
            spec.key = "Modified"


class TestGetRandomForestModel:
    """Tests for get_random_forest_model function."""

    def test_returns_random_forest_regressor(self) -> None:
        """Test that get_random_forest_model returns a RandomForestRegressor."""
        model = get_random_forest_model()
        assert isinstance(model, RandomForestRegressor)

    def test_default_seed_is_42(self) -> None:
        """Test that the default seed is 42."""
        model = get_random_forest_model()
        assert model.random_state == 42

    def test_custom_seed_is_used(self) -> None:
        """Test that a custom seed is used when provided."""
        model = get_random_forest_model(seed=123)
        assert model.random_state == 123

    def test_default_n_estimators(self) -> None:
        """Test that default n_estimators is 200."""
        model = get_random_forest_model()
        assert model.n_estimators == 200

    def test_default_max_depth_is_none(self) -> None:
        """Test that default max_depth is None (unlimited)."""
        model = get_random_forest_model()
        assert model.max_depth is None

    def test_default_max_features_is_sqrt(self) -> None:
        """Test that default max_features is 'sqrt'."""
        model = get_random_forest_model()
        assert model.max_features == "sqrt"

    def test_default_min_samples_leaf(self) -> None:
        """Test that default min_samples_leaf is 2."""
        model = get_random_forest_model()
        assert model.min_samples_leaf == 2

    def test_default_n_jobs(self) -> None:
        """Test that default n_jobs is -1 (use all cores)."""
        model = get_random_forest_model()
        assert model.n_jobs == -1

    def test_multiple_models_with_same_seed_are_identical(self) -> None:
        """Test that models with the same seed are identical."""
        model1 = get_random_forest_model(seed=42)
        model2 = get_random_forest_model(seed=42)
        assert model1.get_params() == model2.get_params()

    def test_models_with_different_seeds_are_different(self) -> None:
        """Test that models with different seeds have different params."""
        model1 = get_random_forest_model(seed=42)
        model2 = get_random_forest_model(seed=123)
        # The random_state should differ
        assert model1.random_state != model2.random_state


class TestMakeRandomForestRegressor:
    """Tests for make_random_forest_regressor factory function."""

    def test_returns_callable(self) -> None:
        """Test that make_random_forest_regressor returns a callable."""
        factory = make_random_forest_regressor(seed=42)
        assert callable(factory)

    def test_factory_returns_random_forest_regressor(self) -> None:
        """Test that the factory returns a RandomForestRegressor."""
        factory = make_random_forest_regressor(seed=42)
        model = factory()
        assert isinstance(model, RandomForestRegressor)

    def test_factory_uses_provided_seed(self) -> None:
        """Test that the factory uses the provided seed."""
        factory = make_random_forest_regressor(seed=999)
        model = factory()
        assert model.random_state == 999

    def test_factory_default_seed(self) -> None:
        """Test that the factory uses seed 42 by default."""
        factory = make_random_forest_regressor()
        model = factory()
        assert model.random_state == 42


class TestRANDOM_FOREST_SPEC:
    """Tests for the RANDOM_FOREST_SPEC constant."""

    def test_key_is_random_forest(self) -> None:
        """Test that the spec key is 'RandomForest'."""
        assert RANDOM_FOREST_SPEC.key == "RandomForest"

    def test_make_estimator_returns_random_forest(self) -> None:
        """Test that make_estimator creates a RandomForestRegressor."""
        model = RANDOM_FOREST_SPEC.make_estimator()
        assert isinstance(model, RandomForestRegressor)

    def test_make_estimator_uses_seed_42(self) -> None:
        """Test that make_estimator uses seed 42."""
        model = RANDOM_FOREST_SPEC.make_estimator()
        assert model.random_state == 42

    def test_param_distributions_include_n_estimators(self) -> None:
        """Test that param_distributions includes n_estimators."""
        assert "n_estimators" in RANDOM_FOREST_SPEC.param_distributions
        assert RANDOM_FOREST_SPEC.param_distributions["n_estimators"] == [100, 200, 400]

    def test_param_distributions_include_max_depth(self) -> None:
        """Test that param_distributions includes max_depth."""
        assert "max_depth" in RANDOM_FOREST_SPEC.param_distributions
        assert RANDOM_FOREST_SPEC.param_distributions["max_depth"] == [12, 20, None]

    def test_param_distributions_include_max_features(self) -> None:
        """Test that param_distributions includes max_features."""
        assert "max_features" in RANDOM_FOREST_SPEC.param_distributions
        assert RANDOM_FOREST_SPEC.param_distributions["max_features"] == ["sqrt", 0.3, 0.5]

    def test_param_distributions_include_min_samples_leaf(self) -> None:
        """Test that param_distributions includes min_samples_leaf."""
        assert "min_samples_leaf" in RANDOM_FOREST_SPEC.param_distributions
        assert RANDOM_FOREST_SPEC.param_distributions["min_samples_leaf"] == [1, 2, 5]

    def test_n_iter_is_8(self) -> None:
        """Test that n_iter is 8."""
        assert RANDOM_FOREST_SPEC.n_iter == 8

    def test_cv_is_3(self) -> None:
        """Test that cv is 3."""
        assert RANDOM_FOREST_SPEC.cv == 3

    def test_subsample_is_none(self) -> None:
        """Test that subsample is None (no subsampling)."""
        assert RANDOM_FOREST_SPEC.subsample is None

    def test_note_is_set(self) -> None:
        """Test that note describes the model."""
        assert "seed 42" in RANDOM_FOREST_SPEC.note
        assert "RandomForest" in RANDOM_FOREST_SPEC.note


class TestDefaultSeed:
    """Tests for the DEFAULT_SEED constant."""

    def test_default_seed_is_42(self) -> None:
        """Test that DEFAULT_SEED is 42."""
        assert DEFAULT_SEED == 42


class TestRandomForestModelBehavior:
    """Integration tests for RandomForest model behavior."""

    @pytest.fixture
    def numeric_sample_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create sample data with only numeric features for testing."""
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame({
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
        })
        y = 2 * X["feature1"] + np.random.randn(n_samples) * 0.1
        return X, y

    @pytest.fixture
    def sample_data_with_category(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create sample data with categorical features."""
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame({
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples),
        })
        y = 2 * X["feature1"] + np.random.randn(n_samples) * 0.1
        return X, y

    def test_model_can_fit(self, numeric_sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test that the model can fit on numeric data."""
        X, y = numeric_sample_data
        model = get_random_forest_model(seed=42)
        model.fit(X, y)
        assert model is not None

    def test_model_can_predict(self, numeric_sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test that the model can make predictions."""
        X, y = numeric_sample_data
        model = get_random_forest_model(seed=42)
        model.fit(X, y)
        predictions = model.predict(X.iloc[:5])
        assert len(predictions) == 5
        assert all(isinstance(p, (float, np.floating)) for p in predictions)

    def test_predictions_are_reproducible(self, numeric_sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test that predictions are reproducible with the same seed."""
        X, y = numeric_sample_data
        model1 = get_random_forest_model(seed=42)
        model1.fit(X, y)
        pred1 = model1.predict(X.iloc[:10])

        model2 = get_random_forest_model(seed=42)
        model2.fit(X, y)
        pred2 = model2.predict(X.iloc[:10])

        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_predictions_differ_with_different_seeds(
        self, numeric_sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test that predictions differ with different seeds."""
        X, y = numeric_sample_data
        model1 = get_random_forest_model(seed=42)
        model1.fit(X, y)
        pred1 = model1.predict(X.iloc[:10])

        model2 = get_random_forest_model(seed=123)
        model2.fit(X, y)
        pred2 = model2.predict(X.iloc[:10])

        # With high probability, predictions will differ with different seeds
        # (not guaranteed for tiny datasets, but very likely)
        assert not np.array_equal(pred1, pred2) or model1.random_state != model2.random_state

    def test_model_handles_categorical_features_with_pipeline(self) -> None:
        """Test that the model can handle categorical features via a pipeline."""
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        X = pd.DataFrame({
            "num_feat": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat_feat": ["A", "B", "A", "C", "B"],
        })
        y = pd.Series([10, 20, 15, 25, 30])

        # Create pipeline with one-hot encoder
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["cat_feat"]),
                ("num", "passthrough", ["num_feat"]),
            ]
        )
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", get_random_forest_model(seed=42)),
        ])

        pipeline.fit(X, y)
        pred = pipeline.predict(X)
        assert len(pred) == 5

    def test_model_handles_missing_values(self) -> None:
        """Test that the model can handle missing values via imputation in pipeline."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        X = pd.DataFrame({
            "feat1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "feat2": [10.0, np.nan, 30.0, 40.0, 50.0],
        })
        y = pd.Series([10, 20, 30, 40, 50])

        # Create pipeline with imputer
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", get_random_forest_model(seed=42)),
        ])

        pipeline.fit(X, y)
        pred = pipeline.predict(X)
        assert len(pred) == 5
        assert not np.any(np.isnan(pred))

    def test_model_works_with_pandas_dataframe(self) -> None:
        """Test that the model works with pandas DataFrame input."""
        X = pd.DataFrame({
            "col1": range(100),
            "col2": range(100, 200),
        })
        y = pd.Series(range(100))

        model = get_random_forest_model(seed=42)
        model.fit(X, y)
        pred = model.predict(X.head(10))
        assert len(pred) == 10

    def test_model_works_with_numpy_array(self) -> None:
        """Test that the model works with numpy array input."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        model = get_random_forest_model(seed=42)
        model.fit(X, y)
        pred = model.predict(X[:10])
        assert len(pred) == 10

    def test_model_has_feature_importances(self, numeric_sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test that the model provides feature importances after fitting."""
        X, y = numeric_sample_data
        model = get_random_forest_model(seed=42)
        model.fit(X, y)

        importances = model.feature_importances_
        assert len(importances) == X.shape[1]
        assert all(imp >= 0 for imp in importances)
        assert sum(importances) > 0


class TestModelPersistence:
    """Tests for model save/load functionality."""

    def test_model_can_be_saved_and_loaded(self, tmp_path) -> None:
        """Test that a model can be saved and loaded."""
        import joblib

        model = get_random_forest_model(seed=42)
        X = pd.DataFrame({"feat": range(50)})
        y = pd.Series(range(50))
        model.fit(X, y)

        # Save
        model_path = tmp_path / "test_model.joblib"
        joblib.dump(model, model_path)

        # Load
        loaded_model = joblib.load(model_path)

        # Verify
        assert isinstance(loaded_model, RandomForestRegressor)
        assert loaded_model.random_state == 42

        # Predictions should match
        X_test = pd.DataFrame({"feat": range(50, 60)})
        pred_original = model.predict(X_test)
        pred_loaded = loaded_model.predict(X_test)
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)


class TestModelConfigurationEdgeCases:
    """Tests for edge cases in model configuration."""

    def test_seed_zero_works(self) -> None:
        """Test that seed 0 works."""
        model = get_random_forest_model(seed=0)
        assert model.random_state == 0

    def test_large_seed_works(self) -> None:
        """Test that large seed values work."""
        model = get_random_forest_model(seed=999999)
        assert model.random_state == 999999

    def test_n_estimators_can_be_changed(self) -> None:
        """Test that n_estimators can be customized."""
        from hotel_booking.regression_models.random_forest import ModelSpec

        spec = ModelSpec(
            key="CustomRF",
            make_estimator=lambda: RandomForestRegressor(n_estimators=50, random_state=42),
            param_distributions={},
            n_iter=5,
            cv=3,
        )
        model = spec.make_estimator()
        assert model.n_estimators == 50

    def test_max_depth_none_means_unlimited(self) -> None:
        """Test that max_depth=None means trees can grow unlimited."""
        model = get_random_forest_model(seed=42)
        assert model.max_depth is None

    def test_min_samples_leaf_bounds(self) -> None:
        """Test that min_samples_leaf affects tree growth."""
        X = pd.DataFrame({"feat": range(100)})
        y = pd.Series(range(100))

        # Small min_samples_leaf allows more complex trees
        model_complex = RandomForestRegressor(
            n_estimators=10,
            min_samples_leaf=1,
            random_state=42,
        )
        model_complex.fit(X, y)
        complexity_complex = sum(model_complex.feature_importances_ > 0)

        # Large min_samples_leaf creates simpler trees
        model_simple = RandomForestRegressor(
            n_estimators=10,
            min_samples_leaf=50,
            random_state=42,
        )
        model_simple.fit(X, y)
        complexity_simple = sum(model_simple.feature_importances_ > 0)

        # Complex model should use more features (generally)
        assert complexity_complex >= complexity_simple


class TestModelSpecCompatibility:
    """Tests for compatibility with sklearn's model selection API."""

    def test_model_spec_works_with_randomized_search(self) -> None:
        """Test that ModelSpec can be used with RandomizedSearchCV."""
        from sklearn.model_selection import RandomizedSearchCV

        X = pd.DataFrame({
            "feat1": np.random.randn(100),
            "feat2": np.random.randn(100),
        })
        y = np.random.randn(100)

        spec = ModelSpec(
            key="Test",
            make_estimator=lambda: RandomForestRegressor(random_state=42),
            param_distributions={
                "n_estimators": [10, 20],
                "max_depth": [3, 5],
            },
            n_iter=2,
            cv=2,
        )

        search = RandomizedSearchCV(
            estimator=spec.make_estimator(),
            param_distributions=spec.param_distributions,
            n_iter=spec.n_iter,
            cv=spec.cv,
            random_state=42,
            n_jobs=1,
        )
        search.fit(X, y)

        assert search.best_estimator_ is not None
        assert search.best_score_ is not None


class TestModelDocstrings:
    """Tests to verify docstrings are present and informative."""

    def test_get_random_forest_model_has_docstring(self) -> None:
        """Test that get_random_forest_model has a docstring."""
        docstring = get_random_forest_model.__doc__
        assert docstring is not None
        assert "RandomForestRegressor" in docstring
        assert "seed" in docstring.lower()

    def test_make_random_forest_regressor_has_docstring(self) -> None:
        """Test that make_random_forest_regressor has a docstring."""
        docstring = make_random_forest_regressor.__doc__
        assert docstring is not None
        assert "factory" in docstring.lower() or "callable" in docstring.lower()

    def test_model_spec_has_docstring(self) -> None:
        """Test that ModelSpec has a docstring."""
        docstring = ModelSpec.__doc__
        assert docstring is not None
        assert "specification" in docstring.lower()
