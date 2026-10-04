"""Classifier mode: optimize a probe set (questions + score + threshold + policy) against a decision model.

Skill mode trains one SKILL.md text, scored rollout by rollout. Classifier mode
trains what a calibrated yes/no model (jev, TypeSafe's System One) needs to
classify well:

    ProbeSet     questions (text, maybe framed), a score over their answers, a threshold,
                 the state budget and the model version the threshold was measured on
    Lab          asks only what the answer cache lacks; everything else is rescored offline
    SplitPlan    group-aware train/val/test splits, held-out groups, a sealed test set
    Objective    recall at a false-positive budget, with constraints, on validation
    ClassifierOptimizer  selection and ensembling over the question pool, then
                 reflective rounds that propose questions, framings and pipeline changes

See docs/classifier.md.
"""

from skill_factory.classifier.cache import AnswerCache
from skill_factory.classifier.dataset import Example, load_examples
from skill_factory.classifier.lab import AnswerTable, Lab, LabSettings
from skill_factory.classifier.objective import Constraint, Objective
from skill_factory.classifier.optimizer import ClassifierOptimizer, ClassifierResult
from skill_factory.classifier.policy import THRESHOLD_POLICY, Policy, load_policy
from skill_factory.classifier.probeset import ProbeSet
from skill_factory.classifier.questions import Question, QuestionBank, QuestionSpec
from skill_factory.classifier.splits import Seal, SealedError, SplitPlan
from skill_factory.classifier.systemone import FakeSystemOne, HttpSystemOne, SystemOneError

__all__ = [
    "AnswerCache",
    "AnswerTable",
    "ClassifierOptimizer",
    "ClassifierResult",
    "Constraint",
    "Example",
    "FakeSystemOne",
    "HttpSystemOne",
    "Lab",
    "LabSettings",
    "Objective",
    "Policy",
    "ProbeSet",
    "Question",
    "QuestionBank",
    "QuestionSpec",
    "Seal",
    "SealedError",
    "SplitPlan",
    "SystemOneError",
    "THRESHOLD_POLICY",
    "load_examples",
    "load_policy",
]
