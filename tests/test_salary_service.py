"""Unit tests for SalaryNormalizer and market tier classification."""

from src.services.salary_service import SalaryNormalizer


def test_classify_market_tier_tier_3():
    offer_de = {"portal": "arbeitnow", "country": "DE", "location": "Berlin"}
    assert SalaryNormalizer.classify_market_tier(offer_de) == "Tier 3 (Europa / Alemania)"

    offer_eu = {"portal": "custom", "country": "ES", "location": "Madrid"}
    assert SalaryNormalizer.classify_market_tier(offer_eu) == "Tier 3 (Europa / Alemania)"


def test_classify_market_tier_tier_2():
    offer_remotive = {"portal": "remotive", "country": "US", "location": "Remote"}
    assert SalaryNormalizer.classify_market_tier(offer_remotive) == "Tier 2 (Tech & Remoto Global)"

    offer_gob = {"portal": "getonboard", "country": "CL", "location": "Remoto en Latinoamérica"}
    assert SalaryNormalizer.classify_market_tier(offer_gob) == "Tier 2 (Tech & Remoto Global)"


def test_classify_market_tier_tier_1():
    offer_local = {"portal": "computrabajo", "country": "PE", "location": "Lima, Perú"}
    assert SalaryNormalizer.classify_market_tier(offer_local) == "Tier 1 (LATAM Local)"


def test_extract_salary_from_structured_bounds():
    offer = {"min_salary_usd": 2000, "max_salary_usd": 4000}
    assert SalaryNormalizer.extract_salary_usd(offer) == 3000.0

    offer_single = {"min_salary_usd": 2500}
    assert SalaryNormalizer.extract_salary_usd(offer_single) == 2500.0


def test_extract_salary_from_pen_text():
    offer = {"salary": "S/. 3,750 al mes"}
    assert SalaryNormalizer.extract_salary_usd(offer) == 1000.0


def test_extract_salary_from_usd_text():
    offer_monthly = {"salary": "$ 2,500 USD"}
    assert SalaryNormalizer.extract_salary_usd(offer_monthly) == 2500.0

    offer_yearly = {"salary": "$60,000 / year"}
    assert SalaryNormalizer.extract_salary_usd(offer_yearly) == 5000.0


def test_extract_salary_from_eur_text():
    offer = {"salary": "48.000 € anual"}
    expected = round((48000.0 / 12.0) * 1.08, 2)
    assert SalaryNormalizer.extract_salary_usd(offer) == expected


def test_extract_salary_unspecified():
    assert SalaryNormalizer.extract_salary_usd({"salary": "A convenir"}) is None
    assert SalaryNormalizer.extract_salary_usd({"salary": "No especificado"}) is None
    assert SalaryNormalizer.extract_salary_usd({}) is None
