"""Tests for strategy selection types."""

import pytest

from evolve_sdk.strategy.types import (
    DetectedParameter,
    ProblemContext,
    StrategyRecommendation,
    StrategyDecision,
    PROBLEM_TYPES,
    STRATEGY_TYPES,
    COMPLEXITY_LEVELS,
)


class TestDetectedParameter:
    """Tests for DetectedParameter dataclass."""

    def test_basic_construction(self):
        param = DetectedParameter(
            name="learning_rate",
            value=0.01,
            location="MySolver.__init__",
        )
        assert param.name == "learning_rate"
        assert param.value == 0.01
        assert param.location == "MySolver.__init__"
        assert param.param_type == "float"
        assert param.description == ""

    def test_full_construction(self):
        param = DetectedParameter(
            name="iterations",
            value=1000,
            location="MySolver.__init__",
            param_type="int",
            description="Number of optimization iterations",
        )
        assert param.param_type == "int"
        assert param.description == "Number of optimization iterations"

    def test_serialization_round_trip(self):
        param = DetectedParameter(
            name="lr",
            value=0.01,
            location="Model.__init__",
            param_type="float",
            description="Learning rate",
        )
        d = param.to_dict()
        restored = DetectedParameter.from_dict(d)
        assert restored.name == param.name
        assert restored.value == param.value
        assert restored.location == param.location
        assert restored.param_type == param.param_type
        assert restored.description == param.description


class TestProblemContext:
    """Tests for ProblemContext dataclass."""

    def test_minimal_construction(self):
        ctx = ProblemContext(
            problem_type="new_algorithm",
            has_parameters=False,
        )
        assert ctx.problem_type == "new_algorithm"
        assert ctx.has_parameters is False
        assert ctx.complexity == "medium"
        assert ctx.similar_past_runs == []
        assert ctx.is_continuation is False

    def test_full_construction(self):
        params = [
            DetectedParameter(name="lr", value=0.01, location="Solver.__init__"),
            DetectedParameter(name="iters", value=100, location="Solver.__init__"),
        ]
        ctx = ProblemContext(
            problem_type="tune_existing",
            has_parameters=True,
            detected_parameters=params,
            complexity="complex",
            similar_past_runs=[{"id": "run_1"}],
            is_continuation=False,
            user_prompt="optimize this solver",
            keywords=["sorting", "algorithm"],
        )
        assert ctx.has_parameters is True
        assert len(ctx.detected_parameters) == 2
        assert ctx.complexity == "complex"
        assert ctx.keywords == ["sorting", "algorithm"]

    def test_all_problem_types_valid(self):
        for pt in PROBLEM_TYPES:
            ctx = ProblemContext(problem_type=pt, has_parameters=False)
            assert ctx.problem_type == pt

    def test_invalid_problem_type_raises(self):
        with pytest.raises(ValueError, match="Invalid problem_type"):
            ProblemContext(problem_type="invalid_type", has_parameters=False)

    def test_invalid_complexity_raises(self):
        with pytest.raises(ValueError, match="Invalid complexity"):
            ProblemContext(
                problem_type="new_algorithm",
                has_parameters=False,
                complexity="extreme",
            )

    def test_serialization_round_trip(self):
        params = [
            DetectedParameter(name="lr", value=0.01, location="Solver.__init__"),
        ]
        ctx = ProblemContext(
            problem_type="tune_existing",
            has_parameters=True,
            detected_parameters=params,
            complexity="simple",
            user_prompt="optimize solver",
            keywords=["optimization"],
        )
        d = ctx.to_dict()
        restored = ProblemContext.from_dict(d)
        assert restored.problem_type == ctx.problem_type
        assert restored.has_parameters == ctx.has_parameters
        assert len(restored.detected_parameters) == 1
        assert restored.detected_parameters[0].name == "lr"
        assert restored.complexity == ctx.complexity
        assert restored.keywords == ctx.keywords


class TestStrategyRecommendation:
    """Tests for StrategyRecommendation dataclass."""

    def test_basic_construction(self):
        rec = StrategyRecommendation(
            strategy="code_evolution",
            confidence=0.85,
            reason="New algorithm with no existing parameters",
        )
        assert rec.strategy == "code_evolution"
        assert rec.confidence == 0.85
        assert rec.memory_enhanced is False

    def test_memory_enhanced(self):
        rec = StrategyRecommendation(
            strategy="hyperparameter_evolution",
            confidence=0.92,
            reason="Historical success on similar problems",
            memory_enhanced=True,
        )
        assert rec.memory_enhanced is True

    def test_all_strategy_types_valid(self):
        for st in STRATEGY_TYPES:
            rec = StrategyRecommendation(
                strategy=st,
                confidence=0.5,
                reason="test",
            )
            assert rec.strategy == st

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Invalid strategy"):
            StrategyRecommendation(
                strategy="magic_evolution",
                confidence=0.5,
                reason="test",
            )

    def test_confidence_too_low_raises(self):
        with pytest.raises(ValueError, match="Confidence must be between"):
            StrategyRecommendation(
                strategy="code_evolution",
                confidence=-0.1,
                reason="test",
            )

    def test_confidence_too_high_raises(self):
        with pytest.raises(ValueError, match="Confidence must be between"):
            StrategyRecommendation(
                strategy="code_evolution",
                confidence=1.5,
                reason="test",
            )

    def test_confidence_boundary_values(self):
        rec_low = StrategyRecommendation(
            strategy="code_evolution", confidence=0.0, reason="test"
        )
        rec_high = StrategyRecommendation(
            strategy="code_evolution", confidence=1.0, reason="test"
        )
        assert rec_low.confidence == 0.0
        assert rec_high.confidence == 1.0

    def test_serialization_round_trip(self):
        rec = StrategyRecommendation(
            strategy="hybrid",
            confidence=0.75,
            reason="Both code and params need work",
            memory_enhanced=True,
        )
        d = rec.to_dict()
        restored = StrategyRecommendation.from_dict(d)
        assert restored.strategy == rec.strategy
        assert restored.confidence == rec.confidence
        assert restored.reason == rec.reason
        assert restored.memory_enhanced == rec.memory_enhanced


class TestStrategyDecision:
    """Tests for StrategyDecision dataclass."""

    def test_basic_construction(self):
        decision = StrategyDecision(strategy="code_evolution")
        assert decision.strategy == "code_evolution"
        assert decision.auto_selected is False
        assert decision.user_confirmed is False
        assert decision.recommendations == []
        assert decision.context is None

    def test_full_construction(self):
        ctx = ProblemContext(problem_type="tune_existing", has_parameters=True)
        recs = [
            StrategyRecommendation(
                strategy="hyperparameter_evolution",
                confidence=0.9,
                reason="Has tunable parameters",
            ),
            StrategyRecommendation(
                strategy="code_evolution",
                confidence=0.4,
                reason="Could also improve code",
            ),
        ]
        decision = StrategyDecision(
            strategy="hyperparameter_evolution",
            auto_selected=False,
            user_confirmed=True,
            recommendations=recs,
            context=ctx,
        )
        assert decision.user_confirmed is True
        assert len(decision.recommendations) == 2
        assert decision.context is not None

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Invalid strategy"):
            StrategyDecision(strategy="bad_strategy")

    def test_serialization_round_trip(self):
        ctx = ProblemContext(
            problem_type="new_algorithm",
            has_parameters=False,
            user_prompt="create sort",
        )
        recs = [
            StrategyRecommendation(
                strategy="code_evolution",
                confidence=0.9,
                reason="New algorithm",
            ),
        ]
        decision = StrategyDecision(
            strategy="code_evolution",
            user_confirmed=True,
            recommendations=recs,
            context=ctx,
        )
        d = decision.to_dict()
        restored = StrategyDecision.from_dict(d)
        assert restored.strategy == decision.strategy
        assert restored.user_confirmed == decision.user_confirmed
        assert len(restored.recommendations) == 1
        assert restored.recommendations[0].strategy == "code_evolution"
        assert restored.context is not None
        assert restored.context.problem_type == "new_algorithm"

    def test_serialization_without_context(self):
        decision = StrategyDecision(strategy="code_evolution")
        d = decision.to_dict()
        restored = StrategyDecision.from_dict(d)
        assert restored.context is None


class TestConstants:
    """Tests for module-level constants."""

    def test_problem_types_complete(self):
        expected = {"new_algorithm", "tune_existing", "stuck_evolution", "hybrid"}
        assert PROBLEM_TYPES == expected

    def test_strategy_types_complete(self):
        expected = {"code_evolution", "hyperparameter_evolution", "hybrid"}
        assert STRATEGY_TYPES == expected

    def test_complexity_levels_complete(self):
        expected = {"simple", "medium", "complex"}
        assert COMPLEXITY_LEVELS == expected

    def test_constants_are_frozen(self):
        with pytest.raises(AttributeError):
            PROBLEM_TYPES.add("new_type")
        with pytest.raises(AttributeError):
            STRATEGY_TYPES.add("new_strategy")
        with pytest.raises(AttributeError):
            COMPLEXITY_LEVELS.add("new_level")
