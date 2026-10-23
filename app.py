from __future__ import annotations

import json
import random
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "notebooks" / "artifacts" / "models" / "nexusiq_logistic_champion_spells.joblib"
METADATA_PATH = PROJECT_ROOT / "notebooks" / "artifacts" / "models" / "nexusiq_model_metadata.json"
CHAMPIONS_PATH = PROJECT_ROOT / "data" / "champs.csv"

EXPECTED_FEATURE_COUNT = 290
EXPECTED_CHAMPION_FEATURES = 272
EXPECTED_SPELL_FEATURES = 18
TEST_ACCURACY = "53.95%"
TEST_ROC_AUC = "0.556"

TEAM_SIZE = 5
SPELLS_PER_PLAYER = 2
ROLES = (
    ("top", "Top"),
    ("jungle", "Jungle"),
    ("mid", "Mid"),
    ("bot", "ADC / Bot"),
    ("support", "Support"),
)
ROLE_KEYS = tuple(role_key for role_key, _ in ROLES)
ROLE_LABELS = dict(ROLES)
# Sensible per-role starting spells (Flash + role staple), using training spell IDs.
ROLE_DEFAULT_SPELLS = {
    "top": (4, 12),      # Flash + Teleport
    "jungle": (4, 11),   # Flash + Smite
    "mid": (4, 14),      # Flash + Ignite
    "bot": (4, 7),       # Flash + Heal
    "support": (4, 14),  # Flash + Ignite
}

DATADRAGON_CDN_ROOT = "https://ddragon.leagueoflegends.com/cdn"
DATADRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DATADRAGON_LOCALE = "en_US"
DATADRAGON_TIMEOUT_SECONDS = 10
EXPECTED_SPELL_NAMES = {
    1: "Cleanse",
    3: "Exhaust",
    4: "Flash",
    6: "Ghost",
    7: "Heal",
    11: "Smite",
    12: "Teleport",
    14: "Ignite",
    21: "Barrier",
}


class NexusIQError(RuntimeError):
    pass


class SchemaError(NexusIQError):
    pass


class DraftValidationError(NexusIQError):
    pass


@dataclass(frozen=True)
class ModelBundle:
    model: object
    feature_names: tuple[str, ...]
    champion_ids: tuple[int, ...]
    spell_ids: tuple[int, ...]
    threshold: float


@dataclass(frozen=True)
class Assets:
    bundle: ModelBundle
    champion_names: dict[int, str]
    champion_options: tuple[int, ...]
    unsupported_champion_ids: tuple[int, ...]


@dataclass(frozen=True)
class TeamDraft:
    champions: tuple[int | None, ...]
    spells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Draft:
    blue: TeamDraft
    red: TeamDraft


@dataclass(frozen=True)
class Prediction:
    blue_probability: float
    red_probability: float
    winner: str
    threshold: float


@dataclass(frozen=True)
class DataDragonMedia:
    version: str
    champion_names: dict[int, str]
    champion_images: dict[int, str]
    champion_tags: dict[int, tuple[str, ...]]
    spell_names: dict[int, str]
    spell_icons: dict[int, str]
    verified_spell_ids: frozenset[int]
    warning: str | None


@dataclass(frozen=True)
class LaneEstimate:
    available: bool
    blue_win_rate: float | None = None
    source: str | None = None
    patch: str | None = None
    region: str | None = None
    rank: str | None = None
    sample_size: int | None = None
    note: str | None = None


def _parse_ids(names: Sequence[str], prefix: str) -> tuple[int, ...]:
    ids: list[int] = []
    for name in names:
        if not name.startswith(prefix):
            raise SchemaError(f"Unexpected feature name '{name}' for prefix '{prefix}'.")
        suffix = name[len(prefix):]
        if not suffix.isdigit():
            raise SchemaError(f"Feature '{name}' does not contain a numeric ID.")
        ids.append(int(suffix))
    return tuple(ids)


def _require_sorted_unique(ids: Sequence[int], label: str) -> None:
    values = tuple(ids)
    if len(values) != len(set(values)) or values != tuple(sorted(values)):
        raise SchemaError(f"{label} features are not in sorted, unique order.")


def _build_bundle(model: object, metadata: dict) -> ModelBundle:
    if not hasattr(model, "predict_proba") or not hasattr(model, "feature_names_in_"):
        raise SchemaError("The saved artifact is not a fitted scikit-learn pipeline with named features.")

    feature_names = tuple(str(name) for name in model.feature_names_in_)
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise SchemaError(
            f"Model expects {len(feature_names)} features, but NexusIQ requires "
            f"exactly {EXPECTED_FEATURE_COUNT}."
        )

    groups = {
        "blue champion": [name for name in feature_names if name.startswith("blue_champ_")],
        "red champion": [name for name in feature_names if name.startswith("red_champ_")],
        "blue spell": [name for name in feature_names if name.startswith("blue_spell_")],
        "red spell": [name for name in feature_names if name.startswith("red_spell_")],
    }

    recognized = sum(len(names) for names in groups.values())
    if recognized != len(feature_names):
        unknown = sorted(set(feature_names) - {name for names in groups.values() for name in names})
        raise SchemaError(f"Unrecognized model features: {', '.join(unknown)}")

    blue_champions = _parse_ids(groups["blue champion"], "blue_champ_")
    red_champions = _parse_ids(groups["red champion"], "red_champ_")
    blue_spells = _parse_ids(groups["blue spell"], "blue_spell_")
    red_spells = _parse_ids(groups["red spell"], "red_spell_")

    _require_sorted_unique(blue_champions, "Blue champion")
    _require_sorted_unique(red_champions, "Red champion")
    _require_sorted_unique(blue_spells, "Blue spell")
    _require_sorted_unique(red_spells, "Red spell")

    if blue_champions != red_champions:
        raise SchemaError("Blue and Red champion feature sets do not match.")
    if blue_spells != red_spells:
        raise SchemaError("Blue and Red spell feature sets do not match.")

    unverified_spells = sorted(set(blue_spells) - set(EXPECTED_SPELL_NAMES))
    if unverified_spells:
        raise SchemaError(
            "The model uses summoner spell IDs without a verified name mapping: "
            f"{unverified_spells}"
        )
    if len(groups["blue champion"]) + len(groups["red champion"]) != EXPECTED_CHAMPION_FEATURES:
        raise SchemaError("The model's champion feature count does not match the training schema.")
    if len(groups["blue spell"]) + len(groups["red spell"]) != EXPECTED_SPELL_FEATURES:
        raise SchemaError("The model's spell feature count does not match the training schema.")

    expected_order = tuple(
        [f"blue_champ_{champion_id}" for champion_id in blue_champions]
        + [f"red_champ_{champion_id}" for champion_id in red_champions]
        + [f"blue_spell_{spell_id}" for spell_id in blue_spells]
        + [f"red_spell_{spell_id}" for spell_id in red_spells]
    )
    if feature_names != expected_order:
        mismatch = next(
            index
            for index, (actual, expected) in enumerate(zip(feature_names, expected_order))
            if actual != expected
        )
        raise SchemaError(
            "Model feature order does not match the validated preprocessing order "
            f"at position {mismatch}: expected '{expected_order[mismatch]}', "
            f"found '{feature_names[mismatch]}'."
        )

    estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    classes = tuple(int(value) for value in getattr(estimator, "classes_", ()))
    if classes != (0, 1):
        raise SchemaError(f"Expected binary classes [0, 1] with Blue win = 1, found {classes}.")

    try:
        threshold = float(metadata["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError("Model metadata does not contain a valid probability threshold.") from exc
    if not 0.0 < threshold < 1.0:
        raise SchemaError(f"Model threshold {threshold} is outside the valid range (0, 1).")
    if metadata.get("target") != "blue_win":
        raise SchemaError(
            f"Expected metadata target 'blue_win', found {metadata.get('target')!r}."
        )

    return ModelBundle(
        model=model,
        feature_names=feature_names,
        champion_ids=blue_champions,
        spell_ids=blue_spells,
        threshold=threshold,
    )


def load_assets() -> Assets:
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise SchemaError(f"Could not load the saved model at {MODEL_PATH}: {exc}") from exc

    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SchemaError(f"Could not load model metadata at {METADATA_PATH}: {exc}") from exc

    try:
        champions = pd.read_csv(CHAMPIONS_PATH)
    except Exception as exc:
        raise SchemaError(f"Could not load champion data at {CHAMPIONS_PATH}: {exc}") from exc

    missing_columns = {"name", "id"} - set(champions.columns)
    if missing_columns:
        raise SchemaError(f"Champion data is missing columns: {sorted(missing_columns)}")

    champions = champions[["name", "id"]].copy()
    champions["name"] = champions["name"].astype(str).str.strip()
    champions["id"] = pd.to_numeric(champions["id"], errors="raise").astype(int)

    if champions["id"].duplicated().any():
        raise SchemaError("Champion data contains duplicate IDs.")
    if (champions["name"] == "").any() or champions["name"].duplicated().any():
        raise SchemaError("Champion data contains blank or duplicate champion names.")

    bundle = _build_bundle(model, metadata)
    catalog_ids = set(champions["id"])
    model_ids = set(bundle.champion_ids)
    missing_from_catalog = model_ids - catalog_ids
    if missing_from_catalog:
        raise SchemaError(
            "The model references champion IDs absent from data/champs.csv: "
            f"{sorted(missing_from_catalog)}"
        )

    champion_names = dict(zip(champions["id"], champions["name"]))
    unsupported = tuple(sorted(catalog_ids - model_ids))
    champion_options = tuple(sorted(bundle.champion_ids, key=lambda champion_id: champion_names[champion_id].lower()))

    return Assets(
        bundle=bundle,
        champion_names=champion_names,
        champion_options=champion_options,
        unsupported_champion_ids=unsupported,
    )


@st.cache_resource(show_spinner="Loading the NexusIQ model...")
def get_assets() -> Assets:
    return load_assets()


def _fallback_spell_name(spell_id: int) -> str:
    return EXPECTED_SPELL_NAMES.get(int(spell_id), f"Spell {int(spell_id)}")


def _fallback_media(spell_ids: Sequence[int], warning: str | None) -> DataDragonMedia:
    return DataDragonMedia(
        version="",
        champion_names={},
        champion_images={},
        champion_tags={},
        spell_names={int(spell_id): _fallback_spell_name(spell_id) for spell_id in spell_ids},
        spell_icons={},
        verified_spell_ids=frozenset(),
        warning=warning,
    )


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_data_dragon_json(url: str) -> object:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "NexusIQ/1.0"},
    )
    with urllib.request.urlopen(request, timeout=DATADRAGON_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"Unexpected Data Dragon payload from {url}.")
    return payload


def _ddragon_image_url(version: str, category: str, file_name: str) -> str:
    return (
        f"{DATADRAGON_CDN_ROOT}/{quote(str(version), safe='')}/img/"
        f"{quote(category, safe='')}/{quote(str(file_name), safe='')}"
    )


@st.cache_data(ttl=3600, show_spinner="Loading Riot Data Dragon artwork...")
def get_data_dragon_media(
    champion_ids: tuple[int, ...],
    spell_ids: tuple[int, ...],
) -> DataDragonMedia:
    wanted_champions = {int(champion_id) for champion_id in champion_ids}
    wanted_spells = {int(spell_id) for spell_id in spell_ids}

    try:
        versions_payload = _fetch_data_dragon_json(DATADRAGON_VERSIONS_URL)
        if not isinstance(versions_payload, list) or not versions_payload:
            raise ValueError("Data Dragon returned no versions.")
        version = str(versions_payload[0]).strip()
        if not version:
            raise ValueError("Data Dragon returned an empty version.")

        champion_payload = _fetch_data_dragon_json(
            f"{DATADRAGON_CDN_ROOT}/{quote(version, safe='')}/data/"
            f"{DATADRAGON_LOCALE}/champion.json"
        )
        spell_payload = _fetch_data_dragon_json(
            f"{DATADRAGON_CDN_ROOT}/{quote(version, safe='')}/data/"
            f"{DATADRAGON_LOCALE}/summoner.json"
        )
        if not isinstance(champion_payload, dict) or not isinstance(spell_payload, dict):
            raise ValueError("Data Dragon metadata is not in the expected format.")

        champion_data = champion_payload.get("data")
        spell_data = spell_payload.get("data")
        if not isinstance(champion_data, dict) or not isinstance(spell_data, dict):
            raise ValueError("Data Dragon metadata is missing its data table.")

        champion_names: dict[int, str] = {}
        champion_images: dict[int, str] = {}
        champion_tags: dict[int, tuple[str, ...]] = {}
        for entry in champion_data.values():
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("key", ""))
            if not key.isdigit() or int(key) not in wanted_champions:
                continue
            champion_id = int(key)
            name = str(entry.get("name") or "").strip()
            image = entry.get("image") if isinstance(entry.get("image"), dict) else {}
            image_file = str(image.get("full") or "").strip()
            tags_payload = entry.get("tags")
            tags = (
                tuple(str(tag).strip() for tag in tags_payload if str(tag).strip())
                if isinstance(tags_payload, list)
                else ()
            )
            if name:
                champion_names[champion_id] = name
            if image_file:
                champion_images[champion_id] = _ddragon_image_url(version, "champion", image_file)
            if tags:
                champion_tags[champion_id] = tags

        spell_names: dict[int, str] = {}
        fetched_spell_names: dict[int, str] = {}
        spell_icons: dict[int, str] = {}
        for entry in spell_data.values():
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("key", ""))
            if not key.isdigit() or int(key) not in wanted_spells:
                continue
            spell_id = int(key)
            name = str(entry.get("name") or "").strip()
            image = entry.get("image") if isinstance(entry.get("image"), dict) else {}
            image_file = str(image.get("full") or "").strip()
            if name:
                spell_names[spell_id] = name
                fetched_spell_names[spell_id] = name
            if image_file:
                spell_icons[spell_id] = _ddragon_image_url(version, "spell", image_file)

        warnings: list[str] = []
        missing_champions = sorted(wanted_champions - set(champion_names) | wanted_champions - set(champion_images))
        if missing_champions:
            warnings.append(
                "Data Dragon did not return complete artwork for supported champion IDs "
                f"{missing_champions}; local names or placeholders are shown for those picks."
            )

        missing_spell_names = sorted(wanted_spells - set(spell_names))
        missing_spell_icons = sorted(wanted_spells - set(spell_icons))
        for spell_id in missing_spell_names:
            spell_names[spell_id] = _fallback_spell_name(spell_id)
        if missing_spell_names:
            warnings.append(
                "Data Dragon did not return names for summoner spell IDs "
                f"{missing_spell_names}; the verified training-ID names are shown."
            )
        if missing_spell_icons:
            warnings.append(
                "Data Dragon did not return icons for summoner spell IDs "
                f"{missing_spell_icons}; text labels remain available."
            )

        verified_spell_ids = frozenset(wanted_spells & set(fetched_spell_names) & set(EXPECTED_SPELL_NAMES))
        name_mismatches = [
            spell_id
            for spell_id in sorted(verified_spell_ids)
            if fetched_spell_names[spell_id] != EXPECTED_SPELL_NAMES[spell_id]
        ]
        if name_mismatches:
            warnings.append(
                "Riot's spell names differ from the expected mapping for IDs "
                f"{name_mismatches}; Riot Data Dragon names are displayed and encoding "
                "continues to use the internal spell IDs."
            )

        return DataDragonMedia(
            version=version,
            champion_names=champion_names,
            champion_images=champion_images,
            champion_tags=champion_tags,
            spell_names=spell_names,
            spell_icons=spell_icons,
            verified_spell_ids=verified_spell_ids,
            warning=" ".join(warnings) or None,
        )
    except Exception as exc:
        return _fallback_media(
            sorted(wanted_spells),
            warning=(
                "Riot Data Dragon artwork could not be loaded "
                f"({type(exc).__name__}: {exc}). Champion names fall back to "
                "data/champs.csv and summoner spells use the verified training-ID "
                "names without icons."
            ),
        )


def _champion_name(assets: Assets, media: DataDragonMedia, champion_id: int) -> str:
    return (
        media.champion_names.get(int(champion_id))
        or assets.champion_names.get(int(champion_id))
        or f"Unknown champion ({int(champion_id)})"
    )


def _spell_name(media: DataDragonMedia, spell_id: int) -> str:
    return media.spell_names.get(int(spell_id)) or _fallback_spell_name(spell_id)


def _champion_options(assets: Assets, media: DataDragonMedia) -> tuple[int, ...]:
    return tuple(
        sorted(
            assets.bundle.champion_ids,
            key=lambda champion_id: _champion_name(assets, media, champion_id).lower(),
        )
    )


def _flatten_spells(team: TeamDraft) -> tuple[int, ...]:
    return tuple(spell for pair in team.spells for spell in pair)


def validate_draft(assets: Assets, draft: Draft) -> None:
    bundle = assets.bundle
    supported_champions = set(bundle.champion_ids)
    supported_spells = set(bundle.spell_ids)
    all_champions: list[int] = []

    for side, team in (("Blue", draft.blue), ("Red", draft.red)):
        if len(team.champions) != TEAM_SIZE:
            raise DraftValidationError(f"{side} side must contain exactly {TEAM_SIZE} champions.")
        if len(team.spells) != TEAM_SIZE:
            raise DraftValidationError(f"{side} side must contain spell selections for all {TEAM_SIZE} players.")

        for champion_id in team.champions:
            if champion_id is None:
                raise DraftValidationError(f"{side} side has an empty champion slot.")
            champion_id = int(champion_id)
            if champion_id not in supported_champions:
                name = assets.champion_names.get(champion_id, "Unknown champion")
                raise DraftValidationError(
                    f"{side} pick {champion_id} ({name}) was not part of the model's training data."
                )
            all_champions.append(champion_id)

        for player_index, pair in enumerate(team.spells, start=1):
            if len(pair) != SPELLS_PER_PLAYER:
                raise DraftValidationError(
                    f"{side} player {player_index} must have exactly {SPELLS_PER_PLAYER} spells."
                )
            for spell_id in pair:
                if spell_id is None or int(spell_id) not in supported_spells:
                    raise DraftValidationError(
                        f"{side} player {player_index} has an unsupported summoner spell ID: {spell_id}."
                    )
            if int(pair[0]) == int(pair[1]):
                raise DraftValidationError(
                    f"{side} player {player_index} has the same summoner spell in both slots."
                )

    duplicate_ids = sorted({champion_id for champion_id in all_champions if all_champions.count(champion_id) > 1})
    if duplicate_ids:
        duplicate_names = [f"{assets.champion_names[champion_id]} ({champion_id})" for champion_id in duplicate_ids]
        raise DraftValidationError(
            "Duplicate champion picks are not allowed: " + ", ".join(duplicate_names) + "."
        )


def build_feature_frame(assets: Assets, draft: Draft) -> pd.DataFrame:
    validate_draft(assets, draft)
    bundle = assets.bundle
    positions = {name: index for index, name in enumerate(bundle.feature_names)}
    row = np.zeros(len(bundle.feature_names), dtype=np.int8)

    for side, team in (("blue", draft.blue), ("red", draft.red)):
        for champion_id in team.champions:
            feature = f"{side}_champ_{int(champion_id)}"
            if feature not in positions:
                raise SchemaError(f"Missing champion feature '{feature}' in the model schema.")
            row[positions[feature]] = 1

        for spell_id in _flatten_spells(team):
            feature = f"{side}_spell_{int(spell_id)}"
            if feature not in positions:
                raise SchemaError(f"Missing spell feature '{feature}' in the model schema.")
            row[positions[feature]] = np.int8(row[positions[feature]] + 1)

    blue_picks = sum(row[positions[f"blue_champ_{champion_id}"]] for champion_id in bundle.champion_ids)
    red_picks = sum(row[positions[f"red_champ_{champion_id}"]] for champion_id in bundle.champion_ids)
    blue_spells = sum(row[positions[f"blue_spell_{spell_id}"]] for spell_id in bundle.spell_ids)
    red_spells = sum(row[positions[f"red_spell_{spell_id}"]] for spell_id in bundle.spell_ids)

    if int(blue_picks) != TEAM_SIZE or int(red_picks) != TEAM_SIZE:
        raise SchemaError("Feature construction produced an invalid number of champion indicators.")
    if int(blue_spells) != TEAM_SIZE * SPELLS_PER_PLAYER or int(red_spells) != TEAM_SIZE * SPELLS_PER_PLAYER:
        raise SchemaError("Feature construction produced an invalid number of spell selections.")

    frame = pd.DataFrame(row.reshape(1, -1), columns=list(bundle.feature_names))
    if tuple(frame.columns) != bundle.feature_names:
        raise SchemaError("Constructed feature columns do not match the model's training order.")
    return frame


def predict_draft(assets: Assets, draft: Draft) -> Prediction:
    frame = build_feature_frame(assets, draft)
    try:
        probabilities = np.asarray(assets.bundle.model.predict_proba(frame), dtype=float)
    except Exception as exc:
        raise NexusIQError(f"Model prediction failed: {exc}") from exc

    if probabilities.shape != (1, 2):
        raise NexusIQError(f"Unexpected prediction shape from the model: {probabilities.shape}.")

    estimator = (
        assets.bundle.model.named_steps.get("model")
        if hasattr(assets.bundle.model, "named_steps")
        else assets.bundle.model
    )
    positive_index = list(estimator.classes_).index(1)
    blue_probability = float(probabilities[0, positive_index])
    if not np.isfinite(blue_probability) or not 0.0 <= blue_probability <= 1.0:
        raise NexusIQError(f"Model returned an invalid Blue win probability: {blue_probability}.")

    threshold = assets.bundle.threshold
    return Prediction(
        blue_probability=blue_probability,
        red_probability=1.0 - blue_probability,
        winner="Blue" if blue_probability >= threshold else "Red",
        threshold=threshold,
    )


def compute_influence(assets: Assets, draft: Draft) -> dict[tuple[str, str], float]:
    """Per-pick model influence, in percentage points toward the pick's own side.

    Each of the ten champion indicators is switched off in turn and the change in the
    model's win probability is recorded. For a linear model this is exactly that
    champion's contribution to the score. It describes the saved model only; it is not
    a lane win rate.
    """
    frame = build_feature_frame(assets, draft)
    bundle = assets.bundle
    positions = {name: index for index, name in enumerate(bundle.feature_names)}
    base_row = frame.iloc[0].to_numpy().copy()

    rows = [base_row]
    slots: list[tuple[str, str]] = []
    for side, team in (("blue", draft.blue), ("red", draft.red)):
        for index, role_key in enumerate(ROLE_KEYS):
            ablated = base_row.copy()
            ablated[positions[f"{side}_champ_{int(team.champions[index])}"]] = 0
            rows.append(ablated)
            slots.append((side, role_key))

    matrix = pd.DataFrame(np.vstack(rows), columns=list(bundle.feature_names))
    try:
        probabilities = np.asarray(bundle.model.predict_proba(matrix), dtype=float)
    except Exception as exc:
        raise NexusIQError(f"Influence analysis failed: {exc}") from exc

    if probabilities.shape != (len(rows), 2):
        raise NexusIQError(
            f"Unexpected influence probability shape from the model: {probabilities.shape}."
        )

    estimator = (
        bundle.model.named_steps.get("model")
        if hasattr(bundle.model, "named_steps")
        else bundle.model
    )
    positive_index = list(estimator.classes_).index(1)
    blue_probs = probabilities[:, positive_index]
    if not np.isfinite(blue_probs).all():
        raise NexusIQError("Influence analysis returned non-finite probabilities.")
    base = float(blue_probs[0])

    influence: dict[tuple[str, str], float] = {}
    for (side, role_key), ablated_probability in zip(slots, blue_probs[1:]):
        delta = (base - float(ablated_probability)) * 100
        influence[(side, role_key)] = delta if side == "blue" else -delta
    return influence


def build_export(
    assets: Assets,
    media: DataDragonMedia,
    draft: Draft,
    prediction: Prediction,
    influence: dict[tuple[str, str], float] | None,
) -> str:
    def team_payload(side: str, team: TeamDraft) -> list[dict]:
        entries = []
        for index, (role_key, role_label) in enumerate(ROLES):
            champion_id = int(team.champions[index])
            entry = {
                "role": role_label,
                "champion": _champion_name(assets, media, champion_id),
                "champion_id": champion_id,
                "spells": [_spell_name(media, int(s)) for s in team.spells[index]],
            }
            if influence and (side, role_key) in influence:
                entry["model_influence_pp"] = round(influence[(side, role_key)], 3)
            entries.append(entry)
        return entries

    payload = {
        "app": "NexusIQ",
        "disclaimer": (
            "Experimental baseline trained on 2016-2017 ranked data. Not validated for "
            "current patches; not competitive advice."
        ),
        "blue": team_payload("blue", draft.blue),
        "red": team_payload("red", draft.red),
        "prediction": {
            "blue_win_probability": round(prediction.blue_probability, 6),
            "red_win_probability": round(prediction.red_probability, 6),
            "predicted_winner": prediction.winner,
            "threshold": prediction.threshold,
        },
        "test_accuracy": TEST_ACCURACY,
        "test_roc_auc": TEST_ROC_AUC,
        "artwork_version": media.version or None,
    }
    return json.dumps(payload, indent=2)


def _role_spell_pair(role_key: str, spell_ids: Sequence[int]) -> tuple[int, int]:
    wanted = ROLE_DEFAULT_SPELLS.get(role_key)
    if wanted and all(spell in set(spell_ids) for spell in wanted):
        return wanted
    return _default_spell_pair(spell_ids)


def _default_spell_pair(spell_ids: Sequence[int]) -> tuple[int, int]:
    values = tuple(spell_ids)
    primary = 4 if 4 in values else values[0]
    secondary = 12 if 12 in values and 12 != primary else next(
        (value for value in values if value != primary),
        primary,
    )
    return primary, secondary


SIDES = ("blue", "red")
CHAMPION_KEY_BY_SLOT = {
    (side, role_key): f"{side}_{role_key}_champion"
    for side in SIDES
    for role_key in ROLE_KEYS
}
SPELL_KEYS_BY_SLOT = {
    (side, role_key): (
        f"{side}_{role_key}_spell_1",
        f"{side}_{role_key}_spell_2",
    )
    for side in SIDES
    for role_key in ROLE_KEYS
}
CHAMPION_KEYS = tuple(
    CHAMPION_KEY_BY_SLOT[(side, role_key)]
    for side in SIDES
    for role_key in ROLE_KEYS
)
SPELL_KEYS = tuple(
    key
    for side in SIDES
    for role_key in ROLE_KEYS
    for key in SPELL_KEYS_BY_SLOT[(side, role_key)]
)
CHAMPION_SLOT_LABELS = {
    CHAMPION_KEY_BY_SLOT[(side, role_key)]: f"{side.title()} {ROLE_LABELS[role_key]}"
    for side in SIDES
    for role_key in ROLE_KEYS
}


def _initialize_widgets(bundle: ModelBundle) -> None:
    for key in CHAMPION_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None
    for side, role_key in CHAMPION_KEY_BY_SLOT:
        primary, secondary = _role_spell_pair(role_key, bundle.spell_ids)
        first_key, second_key = SPELL_KEYS_BY_SLOT[(side, role_key)]
        if first_key not in st.session_state:
            st.session_state[first_key] = primary
        if second_key not in st.session_state:
            st.session_state[second_key] = secondary


def _reset_draft(bundle: ModelBundle) -> None:
    for key in CHAMPION_KEYS:
        st.session_state[key] = None
    for side, role_key in CHAMPION_KEY_BY_SLOT:
        primary, secondary = _role_spell_pair(role_key, bundle.spell_ids)
        first_key, second_key = SPELL_KEYS_BY_SLOT[(side, role_key)]
        st.session_state[first_key] = primary
        st.session_state[second_key] = secondary
    for stale_key in ("prediction", "prediction_signature"):
        st.session_state.pop(stale_key, None)


def _randomize_draft(bundle: ModelBundle) -> None:
    rng = random.SystemRandom()
    picks = rng.sample(list(bundle.champion_ids), TEAM_SIZE * 2)
    for key, champion_id in zip(CHAMPION_KEYS, picks):
        st.session_state[key] = int(champion_id)
    for side, role_key in CHAMPION_KEY_BY_SLOT:
        primary, secondary = _role_spell_pair(role_key, bundle.spell_ids)
        first_key, second_key = SPELL_KEYS_BY_SLOT[(side, role_key)]
        st.session_state[first_key] = primary
        st.session_state[second_key] = secondary
    for stale_key in ("prediction", "prediction_signature"):
        st.session_state.pop(stale_key, None)


def _swap_sides() -> None:
    for role_key in ROLE_KEYS:
        blue_key = CHAMPION_KEY_BY_SLOT[("blue", role_key)]
        red_key = CHAMPION_KEY_BY_SLOT[("red", role_key)]
        st.session_state[blue_key], st.session_state[red_key] = (
            st.session_state.get(red_key),
            st.session_state.get(blue_key),
        )
        for blue_spell, red_spell in zip(
            SPELL_KEYS_BY_SLOT[("blue", role_key)], SPELL_KEYS_BY_SLOT[("red", role_key)]
        ):
            st.session_state[blue_spell], st.session_state[red_spell] = (
                st.session_state.get(red_spell),
                st.session_state.get(blue_spell),
            )
    for stale_key in ("prediction", "prediction_signature"):
        st.session_state.pop(stale_key, None)


def _render_header() -> None:
    st.title("⚔️ NexusIQ")
    st.caption("LEAGUE OF LEGENDS · DRAFT LAB · HISTORICAL MODEL")
    st.write("Build a ten-champion draft, inspect the matchup context, then run the saved model.")

def _render_warning(threshold: float = 0.5) -> None:
    with st.container(border=True):
        st.warning("Experimental historical baseline — not validated for current patches.")
        st.write("Trained on ranked matches from 2016–2017. Treat results as an illustrative baseline, not competitive advice.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{TEST_ACCURACY:.1%}" if isinstance(TEST_ACCURACY, (float, int)) else str(TEST_ACCURACY))
        c2.metric("Test ROC-AUC", f"{TEST_ROC_AUC:.3f}" if isinstance(TEST_ROC_AUC, (float, int)) else str(TEST_ROC_AUC))
        c3.metric("Model features", "290")
        c4.metric("Threshold", f"{threshold:.2f}")

def _champion_snapshot() -> dict[str, int | None]:
    return {key: st.session_state.get(key) for key in CHAMPION_KEYS}


def _available_champion_options(
    slot_key: str,
    champion_options: Sequence[int],
    snapshot: dict[str, int | None],
) -> list[int]:
    taken = {
        int(champion_id)
        for key, champion_id in snapshot.items()
        if key != slot_key and champion_id is not None
    }
    options = [champion_id for champion_id in champion_options if champion_id not in taken]
    current = snapshot.get(slot_key)
    if current is not None and int(current) not in options:
        options.insert(0, int(current))
    return options


def _selection_issues(
    assets: Assets,
    media: DataDragonMedia,
    snapshot: dict[str, int | None],
) -> list[str]:
    supported = set(assets.bundle.champion_ids)
    occurrences: dict[int, list[str]] = {}
    issues: list[str] = []

    for key in CHAMPION_KEYS:
        champion_id = snapshot.get(key)
        if champion_id is None:
            continue
        champion_id = int(champion_id)
        label = CHAMPION_SLOT_LABELS[key]
        if champion_id not in supported:
            issues.append(
                f"{label} contains unsupported champion ID {champion_id}. "
                "Choose a supported champion to continue."
            )
        occurrences.setdefault(champion_id, []).append(label)

    for champion_id, labels in occurrences.items():
        if len(labels) > 1:
            name = _champion_name(assets, media, champion_id)
            issues.append(
                f"Duplicate champion {name} ({champion_id}) is selected in: "
                + ", ".join(labels)
                + "."
            )

    for side in SIDES:
        for role_key in ROLE_KEYS:
            first_key, second_key = SPELL_KEYS_BY_SLOT[(side, role_key)]
            first = st.session_state.get(first_key)
            second = st.session_state.get(second_key)
            if first is not None and first == second:
                issues.append(
                    f"{side.title()} {ROLE_LABELS[role_key]} has the same summoner spell in "
                    f"both slots ({_spell_name(media, int(first))}). Pick two different spells."
                )

    return issues


def _render_champion_portrait(assets: Assets, media: DataDragonMedia, champion_id: int | None) -> None:
    if champion_id is None:
        st.caption("No pick")
        return
    image_url = media.champion_images.get(int(champion_id))
    if image_url:
        st.image(image_url, width=58, caption=_champion_name(assets, media, int(champion_id)))
    else:
        st.caption(_champion_name(assets, media, int(champion_id)))

def _render_spell_icon(media: DataDragonMedia, spell_id: int) -> None:
    icon_url = media.spell_icons.get(int(spell_id))
    if icon_url:
        st.image(icon_url, width=30, caption=_spell_name(media, int(spell_id)))

def _render_team(
    assets: Assets,
    media: DataDragonMedia,
    side: str,
    snapshot: dict[str, int | None],
) -> TeamDraft:
    bundle = assets.bundle
    champion_values: list[int | None] = []
    spell_values: list[tuple[int, int]] = []
    champion_options = _champion_options(assets, media)
    spell_options = tuple(sorted(bundle.spell_ids, key=lambda sid: _spell_name(media, sid).lower()))
    color = "🔵" if side == "blue" else "🔴"
    with st.container(border=True):
        st.subheader(f"{color} {side.title()} side")
        st.caption("Choose one champion and two different summoner spells per role.")
        for role_key, role_label in ROLES:
            champion_key = CHAMPION_KEY_BY_SLOT[(side, role_key)]
            first_spell_key, second_spell_key = SPELL_KEYS_BY_SLOT[(side, role_key)]
            with st.container(border=True):
                st.markdown(f"**{role_label}**")
                champ_col, spell1_col, spell2_col = st.columns([2.3, 1, 1], gap="small")
                with champ_col:
                    selected_champion = st.selectbox(
                        "Champion", options=_available_champion_options(champion_key, champion_options, snapshot),
                        index=None, format_func=lambda cid: _champion_name(assets, media, cid),
                        placeholder="Choose champion", key=champion_key, label_visibility="collapsed",
                    )
                    if selected_champion is not None:
                        image_url = media.champion_images.get(int(selected_champion))
                        if image_url:
                            st.image(image_url, width=70)
                    champion_values.append(selected_champion)
                with spell1_col:
                    first_spell = st.selectbox("Spell 1", options=list(spell_options), format_func=lambda sid: _spell_name(media, sid), key=first_spell_key)
                    icon_url = media.spell_icons.get(int(first_spell))
                    if icon_url:
                        st.image(icon_url, width=34)
                with spell2_col:
                    second_spell = st.selectbox("Spell 2", options=list(spell_options), format_func=lambda sid: _spell_name(media, sid), key=second_spell_key)
                    icon_url = media.spell_icons.get(int(second_spell))
                    if icon_url:
                        st.image(icon_url, width=34)
                spell_values.append((int(first_spell), int(second_spell)))
    return TeamDraft(champions=tuple(champion_values), spells=tuple(spell_values))

def _render_section_header(index: str, title: str, subtitle: str, pill: str | None = None) -> None:
    st.divider()
    st.subheader(f"{index}  ·  {title}")
    st.caption(subtitle)
    if pill:
        st.caption(f"{pill}")

def get_lane_estimate(
    role_key: str,
    blue_champion_id: int | None,
    red_champion_id: int | None,
) -> LaneEstimate:
    role_label = ROLE_LABELS.get(role_key, role_key)
    if blue_champion_id is None or red_champion_id is None:
        return LaneEstimate(
            available=False,
            note=f"Pick both {role_label} champions to review this matchup.",
        )
    return LaneEstimate(
        available=False,
        note=(
            "Lane win rate unavailable. The project has no verified lane-level "
            "outcome dataset or lane model."
        ),
    )


def _render_lane_matchups(
    assets: Assets,
    media: DataDragonMedia,
    blue: TeamDraft,
    red: TeamDraft,
    influence: dict[tuple[str, str], float] | None,
    influence_error: str | None,
) -> None:
    if influence_error:
        st.info(f"Model influence unavailable: {influence_error}")
    if not influence:
        st.info("Complete a valid draft to inspect how individual picks shift the overall model estimate.")
        return
    st.caption("These are changes to the overall Blue-vs-Red model estimate, not lane win rates or champion counters.")
    cols = st.columns(5)
    for i, (role_key, role_label) in enumerate(ROLES):
        blue_id = blue.champions[i]
        red_id = red.champions[i]
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"**{role_label}**")
                if blue_id is not None:
                    st.write(f"🔵 {_champion_name(assets, media, int(blue_id))}")
                else:
                    st.write("🔵 Unpicked")
                if red_id is not None:
                    st.write(f"🔴 {_champion_name(assets, media, int(red_id))}")
                else:
                    st.write("🔴 Unpicked")
                b = influence.get(("blue", role_key))
                r = influence.get(("red", role_key))
                st.metric("Blue pick influence", "—" if b is None else f"{b:+.1f} pp")
                st.metric("Red pick influence", "—" if r is None else f"{r:+.1f} pp")

def _team_tag_counts(
    assets: Assets,
    media: DataDragonMedia,
    team: TeamDraft,
) -> tuple[Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    missing: list[str] = []
    for champion_id in team.champions:
        if champion_id is None:
            continue
        name = _champion_name(assets, media, int(champion_id))
        tags = media.champion_tags.get(int(champion_id), ())
        if tags:
            counts.update(tags)
        else:
            missing.append(name)
    return counts, missing


def _render_draft_summary(
    assets: Assets,
    media: DataDragonMedia,
    draft: Draft,
) -> None:
    left, right = st.columns(2, gap="large")
    for col, side_name, team in ((left, "Blue", draft.blue), (right, "Red", draft.red)):
        with col:
            with st.container(border=True):
                st.markdown(f"### {'🔵' if side_name == 'Blue' else '🔴'} {side_name} draft")
                for i, (role_key, role_label) in enumerate(ROLES):
                    champion_id = team.champions[i]
                    spells = team.spells[i]
                    champion_name = _champion_name(assets, media, int(champion_id)) if champion_id is not None else "Unpicked"
                    spell_names = " · ".join(_spell_name(media, int(spell_id)) for spell_id in spells)
                    st.write(f"**{role_label}:** {champion_name}")
                    st.caption(spell_names)

def _confidence_signal(blue_probability: float) -> tuple[str, float]:
    margin_points = abs(blue_probability - 0.5) * 100
    if margin_points < 2:
        label = "Essentially a coin flip"
    elif margin_points < 7:
        label = "Very narrow output separation"
    elif margin_points < 15:
        label = "Narrow output separation"
    elif margin_points < 25:
        label = "Moderate output separation"
    else:
        label = "Wide output separation"
    return label, margin_points


def _render_prediction(prediction: Prediction) -> None:
    winner_probability = max(prediction.blue_probability, prediction.red_probability)
    confidence_label, margin_points = _confidence_signal(prediction.blue_probability)
    st.success(f"Model output: {prediction.winner} side · {winner_probability:.1%}")
    st.caption("This is the saved historical model's overall match-winner estimate, not a current-patch forecast or lane-level prediction.")
    blue_col, red_col = st.columns(2, gap="large")
    with blue_col:
        st.metric("🔵 Blue win probability", f"{prediction.blue_probability:.1%}")
        st.progress(float(prediction.blue_probability))
    with red_col:
        st.metric("🔴 Red win probability", f"{prediction.red_probability:.1%}")
        st.progress(float(prediction.red_probability))
    with st.container(border=True):
        st.markdown("**How to read this result**")
        st.write(f"Output separation: **{confidence_label}** ({margin_points:.1f} percentage points).")
        st.write(f"The held-out ROC-AUC is {TEST_ROC_AUC}; even a wider gap is weak evidence, not validated certainty.")
    st.caption(f"Prediction rule: Blue wins when Blue probability ≥ {prediction.threshold:.2f}; otherwise Red wins. Probabilities are not calibrated for modern patches.")

def _render_draft_insights(
    assets: Assets,
    media: DataDragonMedia,
    draft: Draft | None,
    blue: TeamDraft,
    red: TeamDraft,
    influence: dict[tuple[str, str], float] | None = None,
) -> None:
    st.caption("Metadata-based observations only. These are not current-patch win rates or independent gameplay predictions.")
    if draft is None:
        st.info("Complete a valid draft to generate insights.")
        return
    blue_counts, blue_missing = _team_tag_counts(assets, media, blue)
    red_counts, red_missing = _team_tag_counts(assets, media, red)
    left, right = st.columns(2, gap="large")
    for col, label, counts in ((left, "🔵 Blue tag profile", blue_counts), (right, "🔴 Red tag profile", red_counts)):
        with col:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                if counts:
                    for tag, count in counts.most_common():
                        st.write(f"{tag}: {count}")
                else:
                    st.caption("No verified tag metadata available.")
    missing_names = sorted(set(blue_missing + red_missing))
    if missing_names:
        st.caption("Tag metadata unavailable for: " + ", ".join(missing_names))
    if influence:
        ranked = sorted(influence.items(), key=lambda item: -abs(item[1]))[:3]
        st.markdown("**Largest model-influence values**")
        for (side, role_key), value in ranked:
            team = blue if side == "blue" else red
            champion_id = team.champions[ROLE_KEYS.index(role_key)]
            name = _champion_name(assets, media, int(champion_id)) if champion_id is not None else "Unpicked"
            st.write(f"{side.title()} {ROLE_LABELS[role_key]} · {name}: {value:+.1f} percentage points")
    st.info("Role and lane labels are not model inputs. No lane advantage, current-patch strength, kills, objectives, or game duration is inferred.")

def _render_model_limitations(bundle: ModelBundle) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training period", "2016–2017")
    c2.metric("Held-out accuracy", f"{TEST_ACCURACY:.1%}" if isinstance(TEST_ACCURACY, (float, int)) else str(TEST_ACCURACY))
    c3.metric("Held-out ROC-AUC", f"{TEST_ROC_AUC:.3f}" if isinstance(TEST_ROC_AUC, (float, int)) else str(TEST_ROC_AUC))
    c4.metric("Threshold", f"{bundle.threshold:.2f}")
    st.markdown("**Important limitations**")
    st.markdown("""
    - Inputs are 272 champion indicators and 18 team-level spell counts (290 features).
    - Role and lane assignments are not model inputs and do not change the overall probability.
    - The model predicts only the overall Blue-vs-Red match winner; it does not predict lane outcomes, kills, objectives, duration, or exact scores.
    - No verified lane-level statistics source is connected, so lane win rates are unavailable.
    - Riot Data Dragon tags are descriptive metadata and may not reflect current balance or optimal roles.
    - The saved artifact is reused unchanged; no retraining, role conditioning, or current-patch validation was performed.
    """)

def main() -> None:
    st.set_page_config(page_title="NexusIQ | Draft Lab", page_icon="⚔️", layout="wide", initial_sidebar_state="collapsed")
    _render_header()
    try:
        assets = get_assets()
    except SchemaError as exc:
        st.error(f"NexusIQ could not start because its model schema could not be verified: {exc}")
        st.stop()
    bundle = assets.bundle
    _render_warning(bundle.threshold)
    media = get_data_dragon_media(bundle.champion_ids, bundle.spell_ids)
    _initialize_widgets(bundle)
    snapshot = _champion_snapshot()

    st.divider()
    st.subheader("01 · Build your draft")
    st.caption("Choose a champion and two different summoner spells for every role. Duplicate champion picks are prevented automatically.")
    action1, action2, action3 = st.columns([1, 1, 4])
    with action1:
        st.button("🎲 Random draft", use_container_width=True, on_click=_randomize_draft, args=(bundle,))
    with action2:
        st.button("⇄ Swap sides", use_container_width=True, on_click=_swap_sides)
    with action3:
        st.button("Reset", on_click=_reset_draft, args=(bundle,))

    blue_col, red_col = st.columns(2, gap="large")
    with blue_col:
        blue = _render_team(assets, media, "blue", snapshot)
    with red_col:
        red = _render_team(assets, media, "red", snapshot)

    support_text = f"{len(bundle.champion_ids)} of {len(assets.champion_names)} champions in data/champs.csv are supported by the saved model."
    if assets.unsupported_champion_ids:
        names = [assets.champion_names[cid] for cid in assets.unsupported_champion_ids]
        support_text += " Unsupported champions are hidden: " + ", ".join(names) + "."
    if media.version:
        support_text += f" Artwork source: Riot Data Dragon {media.version}."
    else:
        support_text += " Riot Data Dragon artwork unavailable; labels remain available."
    st.caption(support_text)
    if media.warning:
        st.warning(media.warning)

    selected_champion_count = sum(1 for side in (blue, red) for champion_id in side.champions if champion_id is not None)
    live_issues = _selection_issues(assets, media, snapshot)
    draft: Draft | None = None
    draft_errors: list[str] = []
    if live_issues:
        status_text = "Fix the draft issues below before predicting."
        draft_errors = live_issues
    elif selected_champion_count < TEAM_SIZE * 2:
        status_text = f"Draft progress: {selected_champion_count}/10 champions selected."
    else:
        try:
            candidate = Draft(blue=blue, red=red)
            validate_draft(assets, candidate)
            draft = candidate
            status_text = "Draft valid · ready to predict."
        except DraftValidationError as exc:
            status_text = "Draft validation failed."
            draft_errors = [str(exc)]
    if draft is not None:
        st.success(status_text)
    elif draft_errors:
        st.error(status_text)
    else:
        st.info(status_text)
    for error in draft_errors:
        st.error(error)

    signature = tuple(st.session_state[key] for key in CHAMPION_KEYS + SPELL_KEYS)
    if st.session_state.get("prediction_signature") != signature:
        for stale_key in ("prediction", "prediction_signature"):
            st.session_state.pop(stale_key, None)
    run_col, _ = st.columns([1, 3])
    with run_col:
        predict_clicked = st.button("⚔️ Predict overall winner", type="primary", use_container_width=True, disabled=draft is None)
    if predict_clicked and draft is not None:
        for stale_key in ("prediction", "prediction_signature"):
            st.session_state.pop(stale_key, None)
        try:
            st.session_state["prediction"] = predict_draft(assets, draft)
            st.session_state["prediction_signature"] = signature
        except NexusIQError as exc:
            for stale_key in ("prediction", "prediction_signature"):
                st.session_state.pop(stale_key, None)
            st.error(f"Prediction failed: {exc}")

    prediction = st.session_state.get("prediction") if st.session_state.get("prediction_signature") == signature else None
    influence: dict[tuple[str, str], float] | None = None
    influence_error: str | None = None
    if draft is not None:
        try:
            influence = compute_influence(assets, draft)
        except NexusIQError as exc:
            influence_error = str(exc)

    with st.expander("02 · Lane-by-lane matchup view", expanded=True):
        _render_lane_matchups(assets, media, blue, red, influence, influence_error)
    with st.expander("03 · Overall prediction", expanded=True):
        if draft is None:
            st.info("Complete a valid draft to see its summary and run the prediction.")
        else:
            _render_draft_summary(assets, media, draft)
            if prediction is not None:
                _render_prediction(prediction)
                st.download_button("Download draft + prediction JSON", data=build_export(assets, media, draft, prediction, influence), file_name="nexusiq_draft_prediction.json", mime="application/json")
            else:
                st.info("Your draft is ready. Click **Predict overall winner** above to calculate the saved model estimate.")
    with st.expander("04 · Draft insights", expanded=False):
        _render_draft_insights(assets, media, draft, blue, red, influence)
    with st.expander("05 · Model details and limitations", expanded=False):
        _render_model_limitations(bundle)
    st.divider()
    st.caption("NexusIQ · experimental historical draft baseline. Not endorsed by Riot Games. League of Legends is a trademark of Riot Games, Inc.")

if __name__ == "__main__":
    main()