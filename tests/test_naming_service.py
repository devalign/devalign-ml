"""Unit tests for ClusterNamingStrategy and taxonomy inference."""

import pytest

from src.services.naming_service import (
    LLMNamingStrategy,
    TaxonomyNamingStrategy,
    get_naming_strategy,
)


@pytest.fixture
def taxonomy_strategy():
    return TaxonomyNamingStrategy()


def test_naming_direct_matches(taxonomy_strategy):
    # FastAPI direct
    assert taxonomy_strategy.name_cluster({"fastapi": 10.0}) == "Desarrollador Backend Python & FastAPI"

    # Terraform direct
    assert taxonomy_strategy.name_cluster({"terraform": 10.0}) == "Ingeniero DevOps & Infraestructura como Código"

    # Flutter direct
    assert taxonomy_strategy.name_cluster({"flutter (software)": 10.0}) == "Desarrollador Mobile Flutter & Dart"

    # Spring Boot direct
    assert taxonomy_strategy.name_cluster({"spring boot": 10.0}) == "Desarrollador Backend Java & Spring Boot"


def test_naming_react_disambiguation(taxonomy_strategy):
    # React + TypeScript
    freq_ts = {"react.js": 10.0, "typescript": 8.0, "next.js": 5.0}
    assert taxonomy_strategy.name_cluster(freq_ts) == "Desarrollador Frontend React & TypeScript"

    # React + React Native
    freq_rn = {"react.js": 10.0, "react native": 8.0}
    assert taxonomy_strategy.name_cluster(freq_rn) == "Desarrollador Mobile React Native"

    # React + Node (MERN)
    freq_mern = {"react.js": 10.0, "node.js": 8.0}
    assert taxonomy_strategy.name_cluster(freq_mern) == "Desarrollador Full Stack MERN (React & Node.js)"


def test_naming_empty_and_fallback(taxonomy_strategy):
    # Empty
    assert taxonomy_strategy.name_cluster({}) == "Especialidad Técnica General"

    # Domain fallback (Vue with no direct rule)
    assert taxonomy_strategy.name_cluster({"vue.js": 10.0}) == "Desarrollador Frontend Vue.js"


def test_factory_get_naming_strategy():
    strat_tax = get_naming_strategy("taxonomy")
    assert isinstance(strat_tax, TaxonomyNamingStrategy)

    strat_llm = get_naming_strategy("llm")
    assert isinstance(strat_llm, LLMNamingStrategy)
