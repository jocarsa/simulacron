#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import datetime as dt
import json
import os
import random
import re
import sqlite3
import sys
import time
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.cm as cm
    import networkx as nx
except ImportError:
    print("Este script requiere requests, matplotlib, networkx. Instálalos con: pip install requests matplotlib networkx", file=sys.stderr)
    sys.exit(1)

# ===== Constants =====
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b-instruct"
REQUEST_TIMEOUT_SECONDS = 1800
SLEEP_BETWEEN_CALLS = 0.8

DEFAULT_TARGETS = {
    "personas": 24,
    "lugares": 16,
    "objetos": 18,
    "emociones": 12,
    "etiquetas": 18,
    "memorias": 120,
    "historial": 18,
}

LIFE_STAGES = [
    ("primera_infancia", 0, 5),
    ("infancia_media", 6, 11),
    ("adolescencia", 12, 17),
    ("juventud_temprana", 18, 24),
    ("adultez_inicial", 25, 34),
    ("adultez_media", 35, 49),
    ("presente", 50, 95),
]

DEFAULT_BATCHES = {
    "entity_passes": 4,
    "memories_per_batch": 16,
    "max_retries_per_call": 4,
}

REL_PRIORITY = {
    "spouse": 100,
    "partner": 95,
    "parent": 90,
    "child": 90,
    "sibling": 80,
    "friend": 60,
}

# Society generation constants
NUM_HOUSES = 64
NUM_WORKPLACES = 6
NUM_SCHOOLS = 2
NUM_HIGHSCHOOLS = 2
NUM_UNIVERSITIES = 2
MIN_PEOPLE_PER_HOUSE = 1
MAX_PEOPLE_PER_HOUSE = 5
MAX_FRIENDS_PER_PERSON = 6
FAMILY_EDGE_TYPES = {"spouse", "partner", "parent", "child", "sibling"}
WEAK_EDGE_TYPES = {"friend"}
LABEL_FONT_SIZE = 6
HOUSE_LAYOUT_SEED = 1234
HOUSE_LAYOUT_K = 2.8
HOUSE_LAYOUT_ITERATIONS = 400
FAMILY_CLUSTER_BASE_RADIUS = 3.6
FAMILY_CLUSTER_RADIUS_STEP = 0.55

MALE_NAMES = [
    "Alejandro", "Álvaro", "Antonio", "Carlos", "Daniel", "David", "Diego",
    "Eduardo", "Enrique", "Fernando", "Francisco", "Gabriel", "Héctor",
    "Ignacio", "Javier", "Jorge", "José", "Juan", "Luis", "Manuel",
    "Marcos", "Mario", "Miguel", "Pablo", "Pedro", "Raúl", "Roberto",
    "Rubén", "Sergio", "Vicente", "Andrés", "Óscar", "Adrián", "Jaime",
    "Iván", "Ricardo", "Tomás", "Salvador", "Emilio", "Víctor"
]

FEMALE_NAMES = [
    "Adriana", "Alba", "Ana", "Beatriz", "Blanca", "Carmen", "Claudia",
    "Cristina", "Elena", "Eva", "Inés", "Irene", "Isabel", "Julia",
    "Laura", "Lucía", "Marta", "María", "Natalia", "Noelia", "Paula",
    "Rocío", "Sara", "Silvia", "Sofía", "Teresa", "Valeria", "Verónica",
    "Andrea", "Patricia", "Alicia", "Raquel", "Pilar", "Nuria", "Lidia",
    "Amparo", "Mónica", "Carla", "Olga", "Esther"
]

SURNAMES = [
    "García", "Martínez", "López", "Sánchez", "Pérez", "Gómez", "Fernández",
    "Ruiz", "Díaz", "Moreno", "Muñoz", "Álvarez", "Romero", "Alonso",
    "Gutiérrez", "Navarro", "Torres", "Domínguez", "Vázquez", "Ramos",
    "Gil", "Ramírez", "Serrano", "Blanco", "Molina", "Morales", "Suárez",
    "Ortega", "Delgado", "Castro", "Ortiz", "Rubio", "Marín", "Sanz",
    "Núñez", "Iglesias", "Medina", "Cortés", "Castillo", "Garrido",
    "Campos", "Vega", "León", "Cano", "Prieto", "Calvo", "Reyes", "Herrera"
]

# ===== Utility Functions =====
def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip(), flags=re.UNICODE)

def sql_escape(value: str) -> str:
    return (value or "").replace("'", "''")

def slugify(text: str) -> str:
    text = normalize_space(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "_", text).strip("_")
    return text or "persona"

def ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]

def safe_int(value: Any, default: int, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        n = int(value)
    except Exception:
        n = default
    if minimum is not None:
        n = max(minimum, n)
    if maximum is not None:
        n = min(maximum, n)
    return n

def unique_strings(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = normalize_space(str(item))
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out

def deep_get(obj: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

# ===== Ollama Client =====
class OllamaClient:
    def __init__(self, url: str, model: str, temperature: float = 0.85):
        self.url = url
        self.model = model
        self.temperature = temperature

    def chat_json(self, system_prompt: str, user_prompt: str, retries: int = 4) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        last_error: Optional[str] = None
        for attempt in range(1, retries + 1):
            try:
                response = requests.post(self.url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                raw = response.json()
                content = deep_get(raw, ["message", "content"], "")
                if not isinstance(content, str) or not content.strip():
                    last_error = f"Respuesta vacía del modelo en intento {attempt}"
                else:
                    data = extract_json_from_text(content)
                    if data is not None:
                        return data
                    last_error = f"No se pudo extraer JSON válido en intento {attempt}"
            except Exception as exc:
                last_error = f"Error en intento {attempt}: {exc}"
            time.sleep(min(8, attempt * 2))
        raise RuntimeError(last_error or "Fallo desconocido al consultar Ollama")

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    code_block = re.search(r"```(?\:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if code_block:
        try:
            data = json.loads(code_block.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return None

# ===== Society Generation Functions =====
def weighted_house_size():
    sizes = [1, 2, 3, 4, 5]
    weights = [0.06, 0.20, 0.28, 0.27, 0.19]
    return random.choices(sizes, weights=weights, k=1)[0]

def random_gender():
    return random.choice(["male", "female"])

def random_first_name(gender):
    return random.choice(MALE_NAMES if gender == "male" else FEMALE_NAMES)

def unique_first_name(used, gender=None):
    for _ in range(200):
        g = gender or random_gender()
        name = random_first_name(g)
        key = (g, name)
        if key not in used:
            used.add(key)
            return g, name
    g = gender or random_gender()
    return g, random_first_name(g)

def random_surname():
    return random.choice(SURNAMES)

def random_surnames_pair():
    s1 = random_surname()
    s2 = random_surname()
    for _ in range(50):
        if s2 != s1:
            break
        s2 = random_surname()
    return s1, s2

def make_person_id(index):
    return f"person_{index:04d}"

def make_house_id(index):
    return f"house_{index:03d}"

def make_workplace_id(index):
    return f"workplace_{index:02d}"

def make_school_id(index):
    return f"school_{index:02d}"

def make_highschool_id(index):
    return f"highschool_{index:02d}"

def make_university_id(index):
    return f"university_{index:02d}"

def age_role(age):
    if age <= 12:
        return "child"
    if age <= 17:
        return "teen"
    if age <= 22:
        return "young_adult"
    if age <= 65:
        return "adult"
    return "retired"

def age_band(age):
    if age <= 12:
        return "child"
    if age <= 17:
        return "teen"
    if age <= 22:
        return "young"
    if age <= 65:
        return "adult"
    return "senior"

def full_name(person):
    parts = [
        (person.get("first_name") or "").strip(),
        (person.get("surname_1") or "").strip(),
        (person.get("surname_2") or "").strip(),
    ]
    return " ".join(p for p in parts if p).strip() or person.get("id", "unknown")

def assign_education_and_work(person):
    person["workplace_id"] = None
    person["school_id"] = None
    person["highschool_id"] = None
    person["university_id"] = None

    age = person["age"]
    if age <= 12:
        person["school_id"] = make_school_id(random.randint(1, NUM_SCHOOLS))
    elif age <= 17:
        person["highschool_id"] = make_highschool_id(random.randint(1, NUM_HIGHSCHOOLS))
    elif age <= 22:
        roll = random.random()
        if roll < 0.70:
            person["university_id"] = make_university_id(random.randint(1, NUM_UNIVERSITIES))
        elif roll < 0.90:
            person["workplace_id"] = make_workplace_id(random.randint(1, NUM_WORKPLACES))
        else:
            person["university_id"] = make_university_id(random.randint(1, NUM_UNIVERSITIES))
            person["workplace_id"] = make_workplace_id(random.randint(1, NUM_WORKPLACES))
    elif age <= 65:
        if random.random() < 0.82:
            person["workplace_id"] = make_workplace_id(random.randint(1, NUM_WORKPLACES))

def add_relation(people_by_id, a_id, b_id, rel_type_a_to_b, rel_type_b_to_a):
    if a_id == b_id:
        return
    a = people_by_id[a_id]
    b = people_by_id[b_id]

    if not any(r["person_id"] == b_id and r["type"] == rel_type_a_to_b for r in a["relations"]):
        a["relations"].append({"person_id": b_id, "type": rel_type_a_to_b})

    if not any(r["person_id"] == a_id and r["type"] == rel_type_b_to_a for r in b["relations"]):
        b["relations"].append({"person_id": a_id, "type": rel_type_b_to_a})

def remove_relation(person, other_id, rel_type=None):
    keep = []
    for r in person["relations"]:
        if r["person_id"] != other_id:
            keep.append(r)
            continue
        if rel_type is not None and r["type"] != rel_type:
            keep.append(r)
    person["relations"] = keep

class PersonFactory:
    def __init__(self):
        self.next_person_index = 1

    def create(self, house_id, age, used_names, gender=None, first_name=None, surname_1=None, surname_2=None):
        g = gender or random_gender()
        if first_name is None:
            g, first_name = unique_first_name(used_names, gender=g)
        pid = make_person_id(self.next_person_index)
        self.next_person_index += 1
        person = {
            "id": pid,
            "house_id": house_id,
            "first_name": first_name,
            "surname_1": surname_1 or random_surname(),
            "surname_2": surname_2 or random_surname(),
            "gender": g,
            "age": int(age),
            "role": age_role(int(age)),
            "workplace_id": None,
            "school_id": None,
            "highschool_id": None,
            "university_id": None,
            "relations": [],
        }
        assign_education_and_work(person)
        return person

def create_child(factory, house_id, used_names, father=None, mother=None, fallback_surname_1=None, fallback_surname_2=None, min_age=0, max_age=17):
    age = random.randint(min_age, max_age)
    surname_1 = father["surname_1"] if father else (fallback_surname_1 or random_surname())
    surname_2 = mother["surname_1"] if mother else (fallback_surname_2 or random_surname())
    return factory.create(
        house_id=house_id,
        age=age,
        used_names=used_names,
        surname_1=surname_1,
        surname_2=surname_2,
    )

def add_sibling_relations(people_by_id, siblings):
    for i in range(len(siblings)):
        for j in range(i + 1, len(siblings)):
            add_relation(people_by_id, siblings[i]["id"], siblings[j]["id"], "sibling", "sibling")

def build_household(factory, house_id, size):
    used_names = set()
    people = []
    logical_family_groups = []

    def register(person):
        people.append(person)
        return person

    if size == 1:
        adult = register(factory.create(house_id, random.randint(18, 90), used_names))
        logical_family_groups.append([adult])
        return people, logical_family_groups

    if size == 2:
        pattern = random.choices(
            ["male_female_pair", "male_male_pair", "female_female_pair"],
            weights=[0.68, 0.16, 0.16],
            k=1,
        )[0]
    else:
        pattern = random.choices(
            [
                "male_female_with_children",
                "male_female_pair",
                "male_male_pair",
                "female_female_pair",
            ],
            weights=[0.74, 0.12, 0.07, 0.07],
            k=1,
        )[0]

    if pattern == "male_male_pair":
        a1 = register(factory.create(house_id, random.randint(22, 80), used_names, gender="male"))
        a2_age = max(18, min(85, a1["age"] + random.randint(-10, 10)))
        a2 = register(factory.create(house_id, a2_age, used_names, gender="male"))
        logical_family_groups.extend([[a1], [a2]])

        while len(people) < size:
            extra = register(factory.create(house_id, random.randint(18, 75), used_names, gender="male"))
            logical_family_groups.append([extra])

    elif pattern == "female_female_pair":
        a1 = register(factory.create(house_id, random.randint(22, 80), used_names, gender="female"))
        a2_age = max(18, min(85, a1["age"] + random.randint(-10, 10)))
        a2 = register(factory.create(house_id, a2_age, used_names, gender="female"))
        logical_family_groups.extend([[a1], [a2]])

        while len(people) < size:
            extra = register(factory.create(house_id, random.randint(18, 75), used_names, gender="female"))
            logical_family_groups.append([extra])

    elif pattern == "male_female_pair":
        male = register(factory.create(house_id, random.randint(22, 80), used_names, gender="male"))
        female_age = max(18, min(85, male["age"] + random.randint(-8, 8)))
        female = register(factory.create(house_id, female_age, used_names, gender="female"))
        logical_family_groups.append([male, female])

    elif pattern == "male_female_with_children":
        father = register(factory.create(house_id, random.randint(26, 55), used_names, gender="male"))
        mother_age = max(18, min(85, father["age"] + random.randint(-7, 7)))
        mother = register(factory.create(house_id, mother_age, used_names, gender="female"))
        children = []
        child_slots = size - 2
        child_max_age = max(0, min(17, min(father["age"], mother["age"]) - 18))
        for _ in range(child_slots):
            child = register(create_child(
                factory,
                house_id,
                used_names,
                father=father,
                mother=mother,
                min_age=0,
                max_age=child_max_age
            ))
            children.append(child)
        logical_family_groups.append([father, mother] + children)

    else:
        for _ in range(size):
            logical_family_groups.append([register(factory.create(house_id, random.randint(18, 75), used_names))])

    return people, logical_family_groups

def realize_family_relations(house_people, family_groups, people_by_id):
    for group in family_groups:
        adults = [p for p in group if p["age"] >= 18]
        minors = [p for p in group if p["age"] < 18]
        young_dependents = [p for p in group if 18 <= p["age"] <= 24]

        couple = None
        explicit_adults = sorted(adults, key=lambda p: p["age"], reverse=True)

        opposite_gender_pairs = []
        for i in range(len(explicit_adults)):
            for j in range(i + 1, len(explicit_adults)):
                a = explicit_adults[i]
                b = explicit_adults[j]
                if abs(a["age"] - b["age"]) <= 12 and a["gender"] != b["gender"]:
                    opposite_gender_pairs.append((abs(a["age"] - b["age"]), a, b))
        if opposite_gender_pairs:
            opposite_gender_pairs.sort(key=lambda x: x[0])
            _, a, b = opposite_gender_pairs[0]
            couple = (a, b)
            rel = "spouse" if random.random() < 0.62 else "partner"
            add_relation(people_by_id, a["id"], b["id"], rel, rel)

        if minors or young_dependents:
            children = sorted(minors + young_dependents, key=lambda p: p["age"])
            parents = []
            if couple:
                parents = [couple[0], couple[1]]
            else:
                adults_by_age = sorted([p for p in adults if p["age"] >= 18], key=lambda p: p["age"], reverse=True)
                if adults_by_age:
                    parents = [adults_by_age[0]]

            for child in children:
                valid_parents = [p for p in parents if p["age"] - child["age"] >= 18]
                if not valid_parents:
                    fallback = [p for p in adults if p["age"] - child["age"] >= 18]
                    valid_parents = fallback[:2]
                for parent in valid_parents:
                    add_relation(people_by_id, parent["id"], child["id"], "parent", "child")

            add_sibling_relations(people_by_id, children)

        elders = [p for p in adults if p["age"] >= 55]
        middle = [p for p in adults if 25 <= p["age"] <= 60]
        for elder in elders:
            for child in middle:
                if elder["id"] == child["id"]:
                    continue
                if elder["age"] - child["age"] >= 18:
                    if not any(r["person_id"] == child["id"] and r["type"] == "parent" for r in elder["relations"]):
                        add_relation(people_by_id, elder["id"], child["id"], "parent", "child")
                    break

    for i in range(len(house_people)):
        for j in range(i + 1, len(house_people)):
            a = house_people[i]
            b = house_people[j]
            if a["age"] >= 18 or b["age"] >= 18:
                continue
            if a["surname_1"] != b["surname_1"] or a["surname_2"] != b["surname_2"]:
                continue
            if abs(a["age"] - b["age"]) <= 12:
                add_relation(people_by_id, a["id"], b["id"], "sibling", "sibling")

def repair_relations(people):
    people_by_id = {p["id"]: p for p in people}

    for p in people:
        dedup = []
        seen = set()
        for r in p["relations"]:
            pid = r.get("person_id")
            rtype = r.get("type")
            if pid not in people_by_id:
                continue
            if pid == p["id"]:
                continue
            key = (pid, rtype)
            if key in seen:
                continue
            seen.add(key)
            dedup.append({"person_id": pid, "type": rtype})
        p["relations"] = dedup

    for p in people:
        for r in list(p["relations"]):
            other = people_by_id[r["person_id"]]
            t = r["type"]

            if t == "parent":
                if p["age"] - other["age"] < 18:
                    remove_relation(p, other["id"], "parent")
                    remove_relation(other, p["id"], "child")

            elif t == "child":
                if other["age"] - p["age"] < 18:
                    remove_relation(p, other["id"], "child")
                    remove_relation(other, p["id"], "parent")

            elif t in ("spouse", "partner"):
                if p["age"] < 18 or other["age"] < 18:
                    remove_relation(p, other["id"], t)
                    remove_relation(other, p["id"], t)

            elif t == "sibling":
                if p["house_id"] != other["house_id"]:
                    remove_relation(p, other["id"], "sibling")
                    remove_relation(other, p["id"], "sibling")

    reverse_map = {
        "parent": "child",
        "child": "parent",
        "spouse": "spouse",
        "partner": "partner",
        "sibling": "sibling",
        "friend": "friend",
    }
    for p in people:
        for r in list(p["relations"]):
            other = people_by_id[r["person_id"]]
            rev = reverse_map.get(r["type"])
            if rev and not any(x["person_id"] == p["id"] and x["type"] == rev for x in other["relations"]):
                other["relations"].append({"person_id": p["id"], "type": rev})

def repair_surnames_from_family(people):
    people_by_id = {p["id"]: p for p in people}

    for person in people:
        parent_ids = [r["person_id"] for r in person["relations"] if r["type"] == "parent"]
        parents = [people_by_id[pid] for pid in parent_ids if pid in people_by_id]
        father = next((p for p in parents if p["gender"] == "male"), None)
        mother = next((p for p in parents if p["gender"] == "female"), None)

        if person["age"] <= 24 and parents:
            if father:
                person["surname_1"] = father["surname_1"]
            elif parents:
                person["surname_1"] = parents[0]["surname_1"]

            if mother:
                person["surname_2"] = mother["surname_1"]
            elif len(parents) >= 2:
                person["surname_2"] = parents[1]["surname_1"]

    for person in people:
        sibling_ids = [r["person_id"] for r in person["relations"] if r["type"] == "sibling"]
        for sid in sibling_ids:
            sib = people_by_id[sid]
            my_parents = {r["person_id"] for r in person["relations"] if r["type"] == "parent"}
            sib_parents = {r["person_id"] for r in sib["relations"] if r["type"] == "parent"}
            if my_parents and my_parents == sib_parents:
                sib["surname_1"] = person["surname_1"]
                sib["surname_2"] = person["surname_2"]

    for person in people:
        if not (person.get("surname_1") or "").strip():
            person["surname_1"] = random_surname()
        if not (person.get("surname_2") or "").strip():
            person["surname_2"] = random_surname()

def ensure_household_adult(people, houses):
    people_by_id = {p["id"]: p for p in people}
    adult_pool = [p for p in people if p["age"] >= 18]

    for house in houses:
        residents = [people_by_id[rid] for rid in house["residents"]]
        if any(r["age"] >= 18 for r in residents):
            continue
        promoted = max(residents, key=lambda p: p["age"])
        promoted["age"] = max(18, promoted["age"])
        promoted["role"] = age_role(promoted["age"])
        assign_education_and_work(promoted)

def assign_friendships(people, people_by_id):
    for person in people:
        desired = random.randint(1, MAX_FRIENDS_PER_PERSON)
        candidates = []

        for other in people:
            if other["id"] == person["id"]:
                continue
            if other["house_id"] == person["house_id"]:
                continue
            if any(r["person_id"] == other["id"] and r["type"] == "friend" for r in person["relations"]):
                continue

            score = 1.0
            if age_band(person["age"]) == age_band(other["age"]):
                score += 4.0
            if person.get("workplace_id") and person.get("workplace_id") == other.get("workplace_id"):
                score += 5.0
            if person.get("school_id") and person.get("school_id") == other.get("school_id"):
                score += 5.0
            if person.get("highschool_id") and person.get("highschool_id") == other.get("highschool_id"):
                score += 5.0
            if person.get("university_id") and person.get("university_id") == other.get("university_id"):
                score += 5.0

            age_diff = abs(person["age"] - other["age"])
            score += max(0.0, 3.0 - age_diff / 8.0)
            if person["role"] == "retired" and other["role"] == "retired":
                score += 1.5

            candidates.append((score, other))

        random.shuffle(candidates)
        candidates.sort(key=lambda x: x[0], reverse=True)

        picks = []
        for score, cand in candidates[:80]:
            if len(picks) >= desired:
                break
            probability = min(0.95, score / 8.2)
            if random.random() < probability:
                picks.append(cand)

        for friend in picks:
            add_relation(people_by_id, person["id"], friend["id"], "friend", "friend")

def cap_friendships(people):
    people_by_id = {p["id"]: p for p in people}

    for p in people:
        friends = [r for r in p["relations"] if r["type"] == "friend"]
        if len(friends) <= MAX_FRIENDS_PER_PERSON:
            continue

        scored = []
        for r in friends:
            other = people_by_id[r["person_id"]]
            score = 0.0
            if age_band(p["age"]) == age_band(other["age"]):
                score += 4.0
            if p.get("workplace_id") and p.get("workplace_id") == other.get("workplace_id"):
                score += 5.0
            if p.get("school_id") and p.get("school_id") == other.get("school_id"):
                score += 5.0
            if p.get("highschool_id") and p.get("highschool_id") == other.get("highschool_id"):
                score += 5.0
            if p.get("university_id") and p.get("university_id") == other.get("university_id"):
                score += 5.0
            score -= abs(p["age"] - other["age"]) / 10.0
            scored.append((score, other["id"]))

        scored.sort(reverse=True)
        keep = {pid for _, pid in scored[:MAX_FRIENDS_PER_PERSON]}
        for r in list(friends):
            if r["person_id"] not in keep:
                remove_relation(p, r["person_id"], "friend")
                remove_relation(people_by_id[r["person_id"]], p["id"], "friend")

def rebuild_full_names(people):
    for p in people:
        p["full_name"] = full_name(p)

def household_childbearing_signature(residents):
    adults = [p for p in residents if p["age"] >= 18]
    adult_males = [p for p in adults if p["gender"] == "male"]
    adult_females = [p for p in adults if p["gender"] == "female"]
    children = [p for p in residents if p["age"] < 18]

    return {
        "adults": adults,
        "adult_males": adult_males,
        "adult_females": adult_females,
        "children": children,
        "can_have_children_under_rule": (
            len(children) > 0 and
            len(adults) == 2 and
            len(adult_males) == 1 and
            len(adult_females) == 1
        )
    }

def validate_census(census):
    errors = []
    warnings = []

    people = census["people"]
    houses = census["houses"]
    people_by_id = {p["id"]: p for p in people}
    houses_by_id = {h["id"]: h for h in houses}

    for p in people:
        if not (p.get("first_name") or "").strip():
            errors.append(f"{p['id']} has empty first_name")
        if not (p.get("surname_1") or "").strip():
            errors.append(f"{p['id']} has empty surname_1")
        if not (p.get("surname_2") or "").strip():
            errors.append(f"{p['id']} has empty surname_2")
        if p["house_id"] not in houses_by_id:
            errors.append(f"{p['id']} points to non-existing house {p['house_id']}")

        age = p.get("age", -1)
        if not isinstance(age, int) or age < 0 or age > 110:
            errors.append(f"{p['id']} has invalid age {age}")

        role_expected = age_role(age)
        if p.get("role") != role_expected:
            warnings.append(f"{p['id']} role mismatch: {p.get('role')} vs {role_expected}")

        if age <= 12:
            if not p.get("school_id"):
                errors.append(f"{p['id']} child without school_id")
            if p.get("workplace_id") or p.get("highschool_id") or p.get("university_id"):
                errors.append(f"{p['id']} child has invalid institution/work")
        elif age <= 17:
            if not p.get("highschool_id"):
                errors.append(f"{p['id']} teen without highschool_id")
            if p.get("school_id") or p.get("university_id"):
                errors.append(f"{p['id']} teen has invalid institution")
        elif age <= 22:
            if not (p.get("university_id") or p.get("workplace_id")):
                warnings.append(f"{p['id']} young adult has neither university nor workplace")
        elif age <= 65:
            if p.get("school_id") or p.get("highschool_id") or p.get("university_id"):
                warnings.append(f"{p['id']} adult has school/highschool/university assignment")
        else:
            if p.get("school_id") or p.get("highschool_id"):
                errors.append(f"{p['id']} retired person has school/highschool assignment")

    seen_residents = set()
    for h in houses:
        if len(h["residents"]) < MIN_PEOPLE_PER_HOUSE:
            errors.append(f"{h['id']} has too few residents")
        if len(h["residents"]) > MAX_PEOPLE_PER_HOUSE:
            errors.append(f"{h['id']} has too many residents")
        if len(set(h["residents"])) != len(h["residents"]):
            errors.append(f"{h['id']} has duplicated resident ids")

        adults = 0
        for rid in h["residents"]:
            if rid not in people_by_id:
                errors.append(f"{h['id']} contains non-existing resident {rid}")
                continue
            if rid in seen_residents:
                errors.append(f"{rid} appears in more than one house")
            seen_residents.add(rid)
            if people_by_id[rid]["house_id"] != h["id"]:
                errors.append(f"{rid} house mismatch between person and house table")
            if people_by_id[rid]["age"] >= 18:
                adults += 1
        if adults < 1:
            errors.append(f"{h['id']} has no adult resident")

        residents = [people_by_id[rid] for rid in h["residents"] if rid in people_by_id]
        signature = household_childbearing_signature(residents)
        if signature["children"]:
            if len(signature["adults"]) != 2:
                errors.append(f"{h['id']} has children but does not have exactly two adults")
            if len(signature["adult_males"]) != 1 or len(signature["adult_females"]) != 1:
                errors.append(f"{h['id']} has children without exactly one adult male and one adult female")

    reverse_map = {
        "parent": "child",
        "child": "parent",
        "spouse": "spouse",
        "partner": "partner",
        "sibling": "sibling",
        "friend": "friend",
    }

    for p in people:
        rel_counter = Counter((r["person_id"], r["type"]) for r in p["relations"])
        for key, count in rel_counter.items():
            if count > 1:
                errors.append(f"{p['id']} has duplicated relation {key}")

        for r in p["relations"]:
            oid = r["person_id"]
            rt = r["type"]
            if oid not in people_by_id:
                errors.append(f"{p['id']} relation points to missing person {oid}")
                continue

            other = people_by_id[oid]
            rev = reverse_map[rt]
            if not any(x["person_id"] == p["id"] and x["type"] == rev for x in other["relations"]):
                errors.append(f"{p['id']} -> {oid} missing reverse relation for {rt}")

            if rt == "parent":
                if p["age"] - other["age"] < 18:
                    errors.append(f"{p['id']} has invalid parent relation with {oid}")
            elif rt == "child":
                if other["age"] - p["age"] < 18:
                    errors.append(f"{p['id']} has invalid child relation with {oid}")
            elif rt in ("spouse", "partner"):
                if p["age"] < 18 or other["age"] < 18:
                    errors.append(f"{p['id']} has underage spouse/partner relation with {oid}")
            elif rt == "sibling":
                if p["house_id"] != other["house_id"]:
                    warnings.append(f"{p['id']} sibling {oid} lives in another house")
                my_parents = {x["person_id"] for x in p["relations"] if x["type"] == "parent"}
                other_parents = {x["person_id"] for x in other["relations"] if x["type"] == "parent"}
                if my_parents and other_parents and my_parents == other_parents:
                    if p["surname_1"] != other["surname_1"] or p["surname_2"] != other["surname_2"]:
                        errors.append(f"{p['id']} and {oid} are siblings with same parents but different surnames")
            elif rt == "friend":
                if p["house_id"] == other["house_id"]:
                    warnings.append(f"{p['id']} friend {oid} is in same house")

    full_name_set = set()
    duplicates = 0
    for p in people:
        fn = full_name(p)
        if fn in full_name_set:
            duplicates += 1
        full_name_set.add(fn)
    if duplicates > len(people) * 0.25:
        warnings.append("High number of duplicate full names")

    return errors, warnings

def generate_census():
    houses = []
    people = []
    people_by_id = {}
    factory = PersonFactory()

    for house_index in range(1, NUM_HOUSES + 1):
        house_id = make_house_id(house_index)
        size = weighted_house_size()
        house_people, family_groups = build_household(factory, house_id, size)

        for p in house_people:
            people.append(p)
            people_by_id[p["id"]] = p

        realize_family_relations(house_people, family_groups, people_by_id)

        houses.append({
            "id": house_id,
            "residents": [p["id"] for p in house_people]
        })

    ensure_household_adult(people, houses)
    repair_relations(people)
    repair_surnames_from_family(people)
    for p in people:
        p["role"] = age_role(p["age"])
        assign_education_and_work(p)

    assign_friendships(people, people_by_id)
    cap_friendships(people)
    repair_relations(people)
    rebuild_full_names(people)

    census = {
        "metadata": {
            "houses": NUM_HOUSES,
            "workplaces": NUM_WORKPLACES,
            "schools": NUM_SCHOOLS,
            "highschools": NUM_HIGHSCHOOLS,
            "universities": NUM_UNIVERSITIES,
            "generated_people": len(people),
        },
        "places": {
            "houses": [{"id": make_house_id(i)} for i in range(1, NUM_HOUSES + 1)],
            "workplaces": [{"id": make_workplace_id(i)} for i in range(1, NUM_WORKPLACES + 1)],
            "schools": [{"id": make_school_id(i)} for i in range(1, NUM_SCHOOLS + 1)],
            "highschools": [{"id": make_highschool_id(i)} for i in range(1, NUM_HIGHSCHOOLS + 1)],
            "universities": [{"id": make_university_id(i)} for i in range(1, NUM_UNIVERSITIES + 1)],
        },
        "houses": houses,
        "people": people,
    }

    errors, warnings = validate_census(census)
    if errors:
        raise RuntimeError(
            "La validación ha fallado:\n" +
            "\n".join(f"- {e}" for e in errors[:100]) +
            ("\n- ... more errors omitted" if len(errors) > 100 else "")
        )

    census["metadata"]["validation_warnings"] = len(warnings)
    census["metadata"]["validation_errors"] = 0
    census["metadata"]["generated_people"] = len(people)

    return census, warnings

# ===== Autobiography Generation Functions =====
def load_census(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def build_indexes(census: Dict[str, Any]) -> Dict[str, Any]:
    people = census.get("people", [])
    houses = census.get("houses", [])
    people_by_id = {p["id"]: p for p in people}
    houses_by_id = {h["id"]: h for h in houses}
    relation_map: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for person in people:
        pid = person["id"]
        for rel in person.get("relations", []):
            other_id = rel.get("person_id")
            rel_type = rel.get("type")
            if other_id in people_by_id and rel_type:
                relation_map[pid][rel_type].append(other_id)
    return {
        "people": people,
        "houses": houses,
        "people_by_id": people_by_id,
        "houses_by_id": houses_by_id,
        "relation_map": relation_map,
    }

def describe_person_brief(person: Dict[str, Any]) -> str:
    gender = "hombre" if person.get("gender") == "male" else "mujer"
    age = person.get("age", 0)
    role = age_role(age)
    role_es = {
        "child": "niño o niña",
        "teen": "adolescente",
        "young_adult": "adulto joven",
        "adult": "adulto",
        "retired": "persona jubilada",
    }.get(role, "persona")
    base = f"{full_name(person)} ({gender}, {age} años, {role_es})"
    if person.get("house_id"):
        base += f", residente en {person['house_id']}"
    place = place_name_from_person(person)
    if place:
        base += f", vinculado a {place}"
    return base

def place_name_from_person(person: Dict[str, Any]) -> Optional[str]:
    for key in ("workplace_id", "university_id", "highschool_id", "school_id"):
        if person.get(key):
            return person[key]
    return None

def collect_social_context(person: Dict[str, Any], idx: Dict[str, Any]) -> Dict[str, Any]:
    people_by_id = idx["people_by_id"]
    relation_map = idx["relation_map"]
    house = idx["houses_by_id"].get(person.get("house_id"), {"residents": []})

    def people_from_ids(ids: List[str]) -> List[Dict[str, Any]]:
        return [people_by_id[x] for x in ids if x in people_by_id]

    spouse = people_from_ids(relation_map[person["id"]].get("spouse", []))
    partner = people_from_ids(relation_map[person["id"]].get("partner", []))
    parents = people_from_ids(relation_map[person["id"]].get("parent", []))
    children = people_from_ids(relation_map[person["id"]].get("child", []))
    siblings = people_from_ids(relation_map[person["id"]].get("sibling", []))
    friends = people_from_ids(relation_map[person["id"]].get("friend", []))
    cohabitants = [people_by_id[rid] for rid in house.get("residents", []) if rid in people_by_id and rid != person["id"]]

    important_people = []
    for rel_type, persons in [
        ("spouse", spouse),
        ("partner", partner),
        ("parent", parents),
        ("child", children),
        ("sibling", siblings),
        ("friend", friends),
    ]:
        for other in persons:
            important_people.append({
                "id": other["id"],
                "nombre": full_name(other),
                "tipo_relacion": rel_type,
                "edad": other.get("age", 0),
                "genero": other.get("gender", ""),
                "descripcion": describe_person_brief(other),
            })
    important_people.sort(key=lambda x: (-REL_PRIORITY.get(x["tipo_relacion"], 0), x["nombre"]))

    entorno_principal = None
    if person.get("workplace_id"):
        entorno_principal = {
            "tipo": "trabajo",
            "id": person["workplace_id"],
            "companeros": [describe_person_brief(p) for p in idx["people"] if p["id"] != person["id"] and p.get("workplace_id") == person["workplace_id"]],
        }
    elif person.get("university_id"):
        entorno_principal = {
            "tipo": "universidad",
            "id": person["university_id"],
            "companeros": [describe_person_brief(p) for p in idx["people"] if p["id"] != person["id"] and p.get("university_id") == person["university_id"]],
        }
    elif person.get("highschool_id"):
        entorno_principal = {
            "tipo": "instituto",
            "id": person["highschool_id"],
            "companeros": [describe_person_brief(p) for p in idx["people"] if p["id"] != person["id"] and p.get("highschool_id") == person["highschool_id"]],
        }
    elif person.get("school_id"):
        entorno_principal = {
            "tipo": "colegio",
            "id": person["school_id"],
            "companeros": [describe_person_brief(p) for p in idx["people"] if p["id"] != person["id"] and p.get("school_id") == person["school_id"]],
        }

    return {
        "persona_central": {
            "id": person["id"],
            "nombre": full_name(person),
            "sexo": person.get("gender", ""),
            "edad": person.get("age", 0),
            "rol": age_role(person.get("age", 0)),
            "casa": person.get("house_id"),
            "trabajo": person.get("workplace_id"),
            "colegio": person.get("school_id"),
            "instituto": person.get("highschool_id"),
            "universidad": person.get("university_id"),
        },
        "pareja": [full_name(p) for p in spouse + partner],
        "padres": [full_name(p) for p in parents],
        "hijos": [full_name(p) for p in children],
        "hermanos": [full_name(p) for p in siblings],
        "amigos": [full_name(p) for p in friends],
        "convivientes": [full_name(p) for p in cohabitants],
        "personas_importantes": important_people[:18],
        "entorno_principal": entorno_principal,
    }

def default_state() -> Dict[str, Any]:
    return {
        "meta": {
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "generator_version": "society-1.0",
        },
        "identity": None,
        "personas": [],
        "lugares": [],
        "objetos": [],
        "emociones": [],
        "etiquetas": [],
        "memorias": [],
        "simulacion_historial": [],
        "social_context": {},
        "source_person": {},
    }

def load_state(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return default_state()
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def save_state(path: Path, state: Dict[str, Any]) -> None:
    state["meta"]["updated_at"] = now_iso()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def unique_by_name(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for item in items:
        key = normalize_space(str(item.get("nombre", ""))).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out

def normalize_entity_item(item: Dict[str, Any], fallback_prefix: str) -> Optional[Dict[str, str]]:
    nombre = normalize_space(str(item.get("nombre", "")))
    descripcion = normalize_space(str(item.get("descripcion", "")))
    if not nombre:
        return None
    if not descripcion:
        descripcion = f"{fallback_prefix}: {nombre}"
    return {"nombre": nombre[:200], "descripcion": descripcion[:2000]}

def merge_entities(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    out = copy.deepcopy(existing)
    existing_keys = {normalize_space(x.get("nombre", "")).lower() for x in out}
    for raw in incoming:
        item = normalize_entity_item(raw, kind)
        if not item:
            continue
        key = item["nombre"].lower()
        if key not in existing_keys:
            out.append(item)
            existing_keys.add(key)
    return out

def normalize_memory_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    titulo = normalize_space(str(item.get("titulo", "")))
    descripcion = normalize_space(str(item.get("descripcion", "")))
    if not titulo or not descripcion:
        return None
    precision_temporal = normalize_space(str(item.get("precision_temporal", "desconocida"))).lower()
    if precision_temporal == "aprox":
        precision_temporal = "aproximada"
    if precision_temporal not in {"exacta", "aproximada", "rango", "desconocida"}:
        precision_temporal = "desconocida"
    return {
        "titulo": titulo[:300],
        "descripcion": descripcion[:5000],
        "fecha_inicio": normalize_space(str(item.get("fecha_inicio", ""))) or None,
        "fecha_fin": normalize_space(str(item.get("fecha_fin", ""))) or None,
        "precision_temporal": precision_temporal,
        "nivel_certeza": safe_int(item.get("nivel_certeza", 3), 3, 1, 5),
        "intensidad_emocional": safe_int(item.get("intensidad_emocional", 3), 3, 1, 5),
        "fuente": normalize_space(str(item.get("fuente", "recuerdo directo")))[:300],
        "notas": normalize_space(str(item.get("notas", "")))[:4000],
        "personas": unique_strings([str(x) for x in ensure_list(item.get("personas"))]),
        "lugares": unique_strings([str(x) for x in ensure_list(item.get("lugares"))]),
        "objetos": unique_strings([str(x) for x in ensure_list(item.get("objetos"))]),
        "emociones": unique_strings([str(x) for x in ensure_list(item.get("emociones"))]),
        "etiquetas": unique_strings([str(x) for x in ensure_list(item.get("etiquetas"))]),
    }

def merge_memories(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = copy.deepcopy(existing)
    keys = {(normalize_space(m.get("titulo", "")).lower(), normalize_space(m.get("descripcion", "")).lower()[:180]) for m in out}
    for raw in incoming:
        item = normalize_memory_item(raw)
        if not item:
            continue
        key = (item["titulo"].lower(), item["descripcion"].lower()[:180])
        if key not in keys:
            out.append(item)
            keys.add(key)
    return out

def absorb_entities_from_memories(state: Dict[str, Any]) -> None:
    catalogs = {k: {x["nombre"].lower() for x in state[k]} for k in ["personas", "lugares", "objetos", "emociones", "etiquetas"]}
    for mem in state["memorias"]:
        for key in ["personas", "lugares", "objetos", "emociones", "etiquetas"]:
            for name in mem.get(key, []):
                low = name.lower()
                if low in catalogs[key]:
                    continue
                state[key].append({"nombre": name, "descripcion": f"Entidad inferida desde memorias: {name}"})
                catalogs[key].add(low)

def ensure_minimum_catalogs(state: Dict[str, Any]) -> None:
    default_emotions = [
        ("Alegría", "Bienestar, entusiasmo o satisfacción."),
        ("Tristeza", "Pena, pérdida o abatimiento."),
        ("Calma", "Serenidad y equilibrio interior."),
        ("Miedo", "Temor, vulnerabilidad o incertidumbre."),
        ("Nostalgia", "Vínculo emocional con el pasado."),
        ("Esperanza", "Confianza moderada en el futuro."),
        ("Vergüenza", "Exposición o incomodidad social."),
        ("Ternura", "Afecto cálido y delicado."),
    ]
    existing = {x["nombre"].lower() for x in state["emociones"]}
    for name, desc in default_emotions:
        if name.lower() not in existing:
            state["emociones"].append({"nombre": name, "descripcion": desc})
    default_tags = [
        ("familia", "Vínculos familiares y escenas de hogar."),
        ("amistad", "Relaciones amistosas."),
        ("hogar", "Recuerdos domésticos y cotidianos."),
        ("trabajo", "Experiencias laborales."),
        ("estudios", "Aprendizaje, colegio, instituto o universidad."),
        ("crecimiento", "Cambios y maduración."),
        ("rutina", "Hábitos y escenas normales."),
        ("barrio", "Vida local y entorno residencial."),
    ]
    existing = {x["nombre"].lower() for x in state["etiquetas"]}
    for name, desc in default_tags:
        if name.lower() not in existing:
            state["etiquetas"].append({"nombre": name, "descripcion": desc})

def sort_entities(state: Dict[str, Any]) -> None:
    for key in ["personas", "lugares", "objetos", "emociones", "etiquetas"]:
        state[key] = sorted(unique_by_name(state[key]), key=lambda x: normalize_space(x["nombre"]).lower())

def sanitize_memory_links(state: Dict[str, Any]) -> None:
    persona_names = {x["nombre"] for x in state["personas"]}
    lugar_names = {x["nombre"] for x in state["lugares"]}
    objeto_names = {x["nombre"] for x in state["objetos"]}
    emocion_names = {x["nombre"] for x in state["emociones"]}
    etiqueta_names = {x["nombre"] for x in state["etiquetas"]}
    for mem in state["memorias"]:
        mem["personas"] = [x for x in unique_strings(mem.get("personas", [])) if x in persona_names]
        mem["lugares"] = [x for x in unique_strings(mem.get("lugares", [])) if x in lugar_names]
        mem["objetos"] = [x for x in unique_strings(mem.get("objetos", [])) if x in objeto_names]
        mem["emociones"] = [x for x in unique_strings(mem.get("emociones", [])) if x in emocion_names]
        mem["etiquetas"] = [x for x in unique_strings(mem.get("etiquetas", [])) if x in etiqueta_names]

def build_global_system_prompt() -> str:
    return """
Eres un generador experto de autobiografías ficticias estructuradas.
Tu tarea es construir la vida interior de una persona dentro de una minisociedad ya existente.
Debes respetar los vínculos sociales dados: parentescos, amistades, convivencia, trabajo y estudios.
Responde SIEMPRE en español y SIEMPRE con JSON válido, sin texto fuera del JSON.
""".strip()

def build_identity_prompt(source_person: Dict[str, Any], social_context: Dict[str, Any], seed_hint: str = "") -> str:
    extra = f"\nReferencia estilística opcional:\n{seed_hint[:5000]}\n" if seed_hint.strip() else ""
    return f"""
Genera la identidad autobiográfica de una persona que ya existe dentro de una sociedad simulada.

Persona base del censo:
{json.dumps(source_person, ensure_ascii=False, indent=2)}

Contexto social y relacional:
{json.dumps(social_context, ensure_ascii=False, indent=2)}

Devuelve JSON con esta estructura exacta:
{{
  "identity": {{
    "nombre": "...",
    "descripcion_corta": "...",
    "biografia_base": "...",
    "valores": "...",
    "estilo_habla": "...",
    "instrucciones_simulacion": "...",
    "edad_actual": 0,
    "anio_nacimiento_aprox": 0,
    "ciudad_origen": "...",
    "pais_origen": "...",
    "rasgos_clave": ["..."],
    "temas_biograficos": ["..."]
  }}
}}

Condiciones:
- El nombre DEBE coincidir con el nombre ya dado en el censo.
- La edad DEBE coincidir con la edad del censo.
- La biografía debe incorporar relaciones reales del contexto social.
- Si estudia o trabaja, debe aparecer como parte de su vida.
{extra}
""".strip()

def build_entities_prompt(identity: Dict[str, Any], source_person: Dict[str, Any], social_context: Dict[str, Any], existing_counts: Dict[str, int], batch_index: int) -> str:
    return f"""
Genera más entidades autobiográficas para enriquecer la base de datos de esta persona.

Identidad actual:
{json.dumps(identity, ensure_ascii=False, indent=2)}

Persona base del censo:
{json.dumps(source_person, ensure_ascii=False, indent=2)}

Contexto social fijo:
{json.dumps(social_context, ensure_ascii=False, indent=2)}

Cantidades ya existentes:
{json.dumps(existing_counts, ensure_ascii=False, indent=2)}

Devuelve JSON exacto con esta forma:
{{
  "personas": [{{"nombre":"...","descripcion":"..."}}],
  "lugares": [{{"nombre":"...","descripcion":"..."}}],
  "objetos": [{{"nombre":"...","descripcion":"..."}}],
  "emociones": [{{"nombre":"...","descripcion":"..."}}],
  "etiquetas": [{{"nombre":"...","descripcion":"..."}}]
}}

Reglas:
- Lote número {batch_index}.
- Debes incluir primero a personas ya existentes en el contexto social si aún no están en el catálogo.
- No cambies los vínculos dados.
""".strip()

def build_memories_prompt(identity: Dict[str, Any], source_person: Dict[str, Any], social_context: Dict[str, Any], stage_name: str, age_from: int, age_to: int, catalog_snapshot: Dict[str, List[str]], existing_count: int, batch_size: int, recent_titles: List[str]) -> str:
    return f"""
Genera {batch_size} memorias autobiográficas nuevas para esta persona.

Etapa vital: {stage_name}
Rango de edad de la etapa: {age_from} a {age_to} años

Identidad:
{json.dumps(identity, ensure_ascii=False, indent=2)}

Persona base del censo:
{json.dumps(source_person, ensure_ascii=False, indent=2)}

Contexto social fijo:
{json.dumps(social_context, ensure_ascii=False, indent=2)}

Catálogo disponible:
{json.dumps(catalog_snapshot, ensure_ascii=False, indent=2)}

Número de memorias ya existentes: {existing_count}
Títulos recientes a evitar:
{json.dumps(recent_titles[-24:], ensure_ascii=False, indent=2)}

Devuelve JSON exacto:
{{
  "memorias": [
    {{
      "titulo": "...",
      "descripcion": "...",
      "fecha_inicio": "YYYY-MM-DD o ''",
      "fecha_fin": "YYYY-MM-DD o ''",
      "precision_temporal": "exacta|aproximada|rango|desconocida",
      "nivel_certeza": 1,
      "intensidad_emocional": 1,
      "fuente": "...",
      "notas": "...",
      "personas": ["..."],
      "lugares": ["..."],
      "objetos": ["..."],
      "emociones": ["..."],
      "etiquetas": ["..."]
    }}
  ]
}}
""".strip()

def build_history_prompt(identity: Dict[str, Any], social_context: Dict[str, Any], sample_memories: List[Dict[str, Any]], desired_pairs: int) -> str:
    compact = []
    for mem in sample_memories[:16]:
        compact.append({
            "titulo": mem.get("titulo"),
            "descripcion": (mem.get("descripcion") or "")[:280],
            "personas": mem.get("personas", [])[:5],
            "lugares": mem.get("lugares", [])[:5],
            "emociones": mem.get("emociones", [])[:5],
            "etiquetas": mem.get("etiquetas", [])[:5],
        })
    return f"""
Genera historial de simulación conversacional de esta persona.

Identidad:
{json.dumps(identity, ensure_ascii=False, indent=2)}

Contexto social:
{json.dumps(social_context, ensure_ascii=False, indent=2)}

Muestra de memorias:
{json.dumps(compact, ensure_ascii=False, indent=2)}

Devuelve JSON exacto:
{{
  "simulacion_historial": [
    {{"role":"system","content":"...","evidencias_json":"[]"}},
    {{"role":"user","content":"...","evidencias_json":"[]"}},
    {{"role":"assistant","content":"...","evidencias_json":"[{{...}}]"}}
  ]
}}

Reglas:
- Aproximadamente {desired_pairs} intercambios user/assistant, más un primer mensaje system.
""".strip()

def generate_identity(client: OllamaClient, state: Dict[str, Any], seed_hint: str = "") -> Dict[str, Any]:
    source_person = state["source_person"]
    social_context = state["social_context"]
    data = client.chat_json(build_global_system_prompt(), build_identity_prompt(source_person, social_context, seed_hint), retries=DEFAULT_BATCHES["max_retries_per_call"])
    identity = data.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("El modelo no devolvió 'identity' correctamente")
    birth_year = dt.datetime.now().year - safe_int(source_person.get("age", 35), 35, 0, 110)
    return {
        "nombre": full_name(source_person),
        "descripcion_corta": normalize_space(str(identity.get("descripcion_corta", ""))),
        "biografia_base": normalize_space(str(identity.get("biografia_base", ""))),
        "valores": normalize_space(str(identity.get("valores", ""))),
        "estilo_habla": normalize_space(str(identity.get("estilo_habla", ""))),
        "instrucciones_simulacion": normalize_space(str(identity.get("instrucciones_simulacion", ""))),
        "edad_actual": safe_int(source_person.get("age", 35), 35, 0, 110),
        "anio_nacimiento_aprox": safe_int(identity.get("anio_nacimiento_aprox", birth_year), birth_year, 1900, dt.datetime.now().year),
        "ciudad_origen": normalize_space(str(identity.get("ciudad_origen", "Valencia"))) or "Valencia",
        "pais_origen": normalize_space(str(identity.get("pais_origen", "España"))) or "España",
        "rasgos_clave": unique_strings([str(x) for x in ensure_list(identity.get("rasgos_clave"))])[:16],
        "temas_biograficos": unique_strings([str(x) for x in ensure_list(identity.get("temas_biograficos"))])[:16],
    }

def bootstrap_catalogs_from_context(state: Dict[str, Any]) -> None:
    social = state["social_context"]
    source = state["source_person"]
    person_catalog = []
    for item in social.get("personas_importantes", []):
        person_catalog.append({
            "nombre": item["nombre"],
            "descripcion": f"Relación {item['tipo_relacion']} de {social['persona_central']['nombre']}. {item['descripcion']}",
        })
    person_catalog.append({"nombre": social["persona_central"]["nombre"], "descripcion": f"Persona central del perfil. Edad {source.get('age', 0)} años."})
    place_catalog = []
    if source.get("house_id"):
        place_catalog.append({"nombre": source["house_id"], "descripcion": "Casa o unidad de residencia principal."})
    for key in ["school_id", "highschool_id", "university_id", "workplace_id"]:
        if source.get(key):
            place_catalog.append({"nombre": source[key], "descripcion": f"Lugar institucional asociado a la persona ({key})."})
    object_catalog = [
        {"nombre": "llaves de casa", "descripcion": "Objeto cotidiano vinculado al hogar."},
        {"nombre": "móvil personal", "descripcion": "Objeto cotidiano de comunicación y rutina."},
        {"nombre": "ropa de diario", "descripcion": "Prendas habituales de la vida diaria."},
    ]
    state["personas"] = merge_entities(state["personas"], person_catalog, "persona")
    state["lugares"] = merge_entities(state["lugares"], place_catalog, "lugar")
    state["objetos"] = merge_entities(state["objetos"], object_catalog, "objeto")
    ensure_minimum_catalogs(state)

def generate_entities_pass(client: OllamaClient, state: Dict[str, Any], batch_index: int) -> None:
    payload = client.chat_json(
        build_global_system_prompt(),
        build_entities_prompt(
            state["identity"],
            state["source_person"],
            state["social_context"],
            {
                "personas": len(state["personas"]),
                "lugares": len(state["lugares"]),
                "objetos": len(state["objetos"]),
                "emociones": len(state["emociones"]),
                "etiquetas": len(state["etiquetas"]),
            },
            batch_index,
        ),
        retries=DEFAULT_BATCHES["max_retries_per_call"],
    )
    state["personas"] = merge_entities(state["personas"], ensure_list(payload.get("personas")), "persona")
    state["lugares"] = merge_entities(state["lugares"], ensure_list(payload.get("lugares")), "lugar")
    state["objetos"] = merge_entities(state["objetos"], ensure_list(payload.get("objetos")), "objeto")
    state["emociones"] = merge_entities(state["emociones"], ensure_list(payload.get("emociones")), "emocion")
    state["etiquetas"] = merge_entities(state["etiquetas"], ensure_list(payload.get("etiquetas")), "etiqueta")

def catalog_snapshot(state: Dict[str, Any]) -> Dict[str, List[str]]:
    return {key: [x["nombre"] for x in state[key]] for key in ["personas", "lugares", "objetos", "emociones", "etiquetas"]}

def generate_memories_batch(client: OllamaClient, state: Dict[str, Any], stage_name: str, age_from: int, age_to: int, batch_size: int) -> None:
    recent_titles = [m.get("titulo", "") for m in state["memorias"]]
    payload = client.chat_json(
        build_global_system_prompt(),
        build_memories_prompt(state["identity"], state["source_person"], state["social_context"], stage_name, age_from, age_to, catalog_snapshot(state), len(state["memorias"]), batch_size, recent_titles),
        retries=DEFAULT_BATCHES["max_retries_per_call"],
    )
    state["memorias"] = merge_memories(state["memorias"], ensure_list(payload.get("memorias")))
    absorb_entities_from_memories(state)

def generate_history(client: OllamaClient, state: Dict[str, Any], desired_pairs: int) -> None:
    memories = state["memorias"][:]
    random.shuffle(memories)
    payload = client.chat_json(build_global_system_prompt(), build_history_prompt(state["identity"], state["social_context"], memories, desired_pairs), retries=DEFAULT_BATCHES["max_retries_per_call"])
    cleaned = []
    for item in ensure_list(payload.get("simulacion_historial")):
        if not isinstance(item, dict):
            continue
        role = normalize_space(str(item.get("role", ""))).lower()
        content = normalize_space(str(item.get("content", "")))
        evidencias_json = str(item.get("evidencias_json", "[]")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        try:
            json.loads(evidencias_json)
        except Exception:
            evidencias_json = "[]"
        cleaned.append({"role": role, "content": content, "evidencias_json": evidencias_json})
    state["simulacion_historial"] = cleaned

def assign_ids(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for idx, item in enumerate(items, start=1):
        clone = copy.deepcopy(item)
        clone["id"] = idx
        out.append(clone)
    return out

def export_sql(state: Dict[str, Any], output_path: Path) -> None:
    ensure_minimum_catalogs(state)
    sort_entities(state)
    sanitize_memory_links(state)
    personas = assign_ids(state["personas"])
    lugares = assign_ids(state["lugares"])
    objetos = assign_ids(state["objetos"])
    emociones = assign_ids(state["emociones"])
    etiquetas = assign_ids(state["etiquetas"])
    memorias = assign_ids(state["memorias"])
    persona_id = {x["nombre"]: x["id"] for x in personas}
    lugar_id = {x["nombre"]: x["id"] for x in lugares}
    objeto_id = {x["nombre"]: x["id"] for x in objetos}
    emocion_id = {x["nombre"]: x["id"] for x in emociones}
    etiqueta_id = {x["nombre"]: x["id"] for x in etiquetas}
    identity = state["identity"] or {}
    lines: List[str] = []
    lines.append("-- SEMILLA GENERADA AUTOMÁTICAMENTE PARA UNA PERSONA DE LA SOCIEDAD")
    lines.append(f"-- Fecha: {now_iso()}")
    lines.append(f"-- Persona: {identity.get('nombre', 'Persona simulada')}")
    lines.append("")
    lines.append("PRAGMA foreign_keys = ON;")
    lines.append("BEGIN TRANSACTION;")
    lines.append("")
    lines.append("DELETE FROM simulacion_historial;")
    lines.append("DELETE FROM memoria_etiqueta;")
    lines.append("DELETE FROM memoria_emocion;")
    lines.append("DELETE FROM memoria_objeto;")
    lines.append("DELETE FROM memoria_lugar;")
    lines.append("DELETE FROM memoria_persona;")
    lines.append("DELETE FROM memorias;")
    lines.append("DELETE FROM etiquetas;")
    lines.append("DELETE FROM emociones;")
    lines.append("DELETE FROM objetos;")
    lines.append("DELETE FROM lugares;")
    lines.append("DELETE FROM personas;")
    lines.append("DELETE FROM perfil_identidad;")
    lines.append("")

    def emit_table(table_name: str, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            lines.append(f"INSERT INTO {table_name} (id, nombre, descripcion, created_at) VALUES ({row['id']}, '{sql_escape(row['nombre'])}', '{sql_escape(row['descripcion'])}', datetime('now'));")
        lines.append("")

    emit_table("personas", personas)
    emit_table("lugares", lugares)
    emit_table("objetos", objetos)
    emit_table("emociones", emociones)
    emit_table("etiquetas", etiquetas)

    for mem in memorias:
        fecha_inicio = f"'{sql_escape(mem['fecha_inicio'])}'" if mem.get("fecha_inicio") else "NULL"
        fecha_fin = f"'{sql_escape(mem['fecha_fin'])}'" if mem.get("fecha_fin") else "NULL"
        lines.append(
            "INSERT INTO memorias (id, titulo, descripcion, fecha_inicio, fecha_fin, precision_temporal, nivel_certeza, intensidad_emocional, fuente, notas, created_at, updated_at) VALUES ("
            f"{mem['id']}, '{sql_escape(mem['titulo'])}', '{sql_escape(mem['descripcion'])}', {fecha_inicio}, {fecha_fin}, '{sql_escape(mem['precision_temporal'])}', {safe_int(mem['nivel_certeza'], 3, 1, 5)}, {safe_int(mem['intensidad_emocional'], 3, 1, 5)}, '{sql_escape(mem.get('fuente', 'recuerdo directo'))}', '{sql_escape(mem.get('notas', ''))}', datetime('now'), datetime('now'));"
        )
    lines.append("")

    def emit_rel(table_name: str, memory_key: str, mapping: Dict[str, int], column: str) -> None:
        for mem in memorias:
            for name in unique_strings(mem.get(memory_key, [])):
                rel_id = mapping.get(name)
                if rel_id is not None:
                    lines.append(f"INSERT INTO {table_name} (memoria_id, {column}) VALUES ({mem['id']}, {rel_id});")
        lines.append("")

    emit_rel("memoria_persona", "personas", persona_id, "persona_id")
    emit_rel("memoria_lugar", "lugares", lugar_id, "lugar_id")
    emit_rel("memoria_objeto", "objetos", objeto_id, "objeto_id")
    emit_rel("memoria_emocion", "emociones", emocion_id, "emocion_id")
    emit_rel("memoria_etiqueta", "etiquetas", etiqueta_id, "etiqueta_id")

    for item in state["simulacion_historial"]:
        lines.append(f"INSERT INTO simulacion_historial (role, content, evidencias_json, created_at) VALUES ('{sql_escape(item.get('role', 'user'))}', '{sql_escape(item.get('content', ''))}', '{sql_escape(item.get('evidencias_json', '[]'))}', datetime('now'));")
    lines.append("")
    lines.append("COMMIT;")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

def print_progress(person_name: str, state: Dict[str, Any], targets: Dict[str, int]) -> None:
    print(
        f"[{now_iso()}] {person_name} | "
        f"personas={len(state['personas'])}/{targets['personas']} | "
        f"lugares={len(state['lugares'])}/{targets['lugares']} | "
        f"objetos={len(state['objetos'])}/{targets['objetos']} | "
        f"emociones={len(state['emociones'])}/{targets['emociones']} | "
        f"etiquetas={len(state['etiquetas'])}/{targets['etiquetas']} | "
        f"memorias={len(state['memorias'])}/{targets['memorias']} | "
        f"historial={len(state['simulacion_historial'])}/{targets['historial']}"
    )

def build_person_dir(base_dir: Path, person: Dict[str, Any]) -> Path:
    return base_dir / f"{person['id']}_{slugify(full_name(person))}"

def generate_for_person(client: OllamaClient, person: Dict[str, Any], social_context: Dict[str, Any], out_dir: Path, seed_hint: str, targets: Dict[str, int], entity_passes: int, memories_per_batch: int, force_history: bool = False) -> None:
    person_dir = build_person_dir(out_dir, person)
    person_dir.mkdir(parents=True, exist_ok=True)
    state_path = person_dir / "state.json"
    sql_path = person_dir / "seed.sql"
    context_path = person_dir / "prompt_context.json"
    state = load_state(state_path)
    state["source_person"] = {
        "id": person["id"],
        "full_name": full_name(person),
        "first_name": person.get("first_name", ""),
        "surname_1": person.get("surname_1", ""),
        "surname_2": person.get("surname_2", ""),
        "gender": person.get("gender", ""),
        "age": person.get("age", 0),
        "role": person.get("role", age_role(person.get("age", 0))),
        "house_id": person.get("house_id"),
        "school_id": person.get("school_id"),
        "highschool_id": person.get("highschool_id"),
        "university_id": person.get("university_id"),
        "workplace_id": person.get("workplace_id"),
    }
    state["social_context"] = social_context
    with open(context_path, "w", encoding="utf-8") as fh:
        json.dump({"source_person": state["source_person"], "social_context": social_context}, fh, ensure_ascii=False, indent=2)

    name = full_name(person)
    if state.get("identity") is None:
        print(f"[INFO] Generando identidad para {name}...")
        state["identity"] = generate_identity(client, state, seed_hint=seed_hint)
        save_state(state_path, state)

    bootstrap_catalogs_from_context(state)
    ensure_minimum_catalogs(state)
    save_state(state_path, state)
    print_progress(name, state, targets)

    for batch in range(1, entity_passes + 1):
        if len(state["personas"]) >= targets["personas"] and len(state["lugares"]) >= targets["lugares"] and len(state["objetos"]) >= targets["objetos"] and len(state["emociones"]) >= targets["emociones"] and len(state["etiquetas"]) >= targets["etiquetas"]:
            break
        print(f"[INFO] Entidades {name}, lote {batch}...")
        generate_entities_pass(client, state, batch)
        ensure_minimum_catalogs(state)
        save_state(state_path, state)
        print_progress(name, state, targets)
        time.sleep(SLEEP_BETWEEN_CALLS)

    cycle = 0
    while len(state["memorias"]) < targets["memorias"] and cycle < 99999:
        cycle += 1
        for stage_name, age_from, age_to in LIFE_STAGES:
            current_age = safe_int(person.get("age", 0), 0, 0, 110)
            if age_from > current_age:
                continue
            stage_to = min(age_to, current_age)
            if stage_to < age_from:
                continue
            if len(state["memorias"]) >= targets["memorias"]:
                break
            print(f"[INFO] Memorias {name}: etapa={stage_name}, ciclo={cycle}")
            before = len(state["memorias"])
            generate_memories_batch(client, state, stage_name, age_from, stage_to, memories_per_batch)
            ensure_minimum_catalogs(state)
            after = len(state["memorias"])
            save_state(state_path, state)
            print(f"[OK] {name}: memorias añadidas {after - before}")
            print_progress(name, state, targets)
            time.sleep(SLEEP_BETWEEN_CALLS)

    if force_history or len(state["simulacion_historial"]) < targets["historial"]:
        print(f"[INFO] Historial conversacional para {name}...")
        generate_history(client, state, targets["historial"])
        save_state(state_path, state)
        print_progress(name, state, targets)

    print(f"[INFO] Exportando SQL para {name}...")
    export_sql(state, sql_path)
    print(f"[OK] SQL guardado: {sql_path}")

def read_optional_text(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return ""
    with open(p, "r", encoding="utf-8") as fh:
        return fh.read()

def auto_detect_census(cli_value: str) -> str:
    if cli_value and Path(cli_value).is_file():
        return cli_value
    candidates = [
        Path("census.json"),
        Path("./census.json"),
        Path("./salida/census.json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    json_candidates = sorted(Path(".").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in json_candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "people" in data and "houses" in data:
                return str(candidate)
        except Exception:
            continue
    raise FileNotFoundError("No encuentro un census.json válido. Pon el archivo en la carpeta actual o usa --census ruta/al/archivo.json")

def auto_detect_seed_hint(cli_value: str) -> str:
    if cli_value and Path(cli_value).is_file():
        return cli_value
    candidates = [
        Path("semilla_persona_simulada.sql"),
        Path("schema.sql"),
        Path("./generador/semilla_persona_simulada.sql"),
        Path("./generador/schema.sql"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""

def select_people(all_people: List[Dict[str, Any]], person_ids: List[str], max_people: Optional[int]) -> List[Dict[str, Any]]:
    selected = all_people
    if person_ids:
        wanted = set(person_ids)
        selected = [p for p in all_people if p["id"] in wanted]
    if max_people is not None:
        selected = selected[:max_people]
    return selected

# ===== Database Functions =====
def create_database(db_path: str, census: Dict[str, Any], people_data: List[Dict[str, Any]]) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS houses (
        id TEXT PRIMARY KEY,
        residents_count INTEGER
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS people (
        id TEXT PRIMARY KEY,
        full_name TEXT,
        first_name TEXT,
        surname_1 TEXT,
        surname_2 TEXT,
        gender TEXT,
        age INTEGER,
        role TEXT,
        house_id TEXT,
        school_id TEXT,
        highschool_id TEXT,
        university_id TEXT,
        workplace_id TEXT,
        FOREIGN KEY (house_id) REFERENCES houses(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS relations (
        person_id TEXT,
        other_person_id TEXT,
        relation_type TEXT,
        PRIMARY KEY (person_id, other_person_id, relation_type),
        FOREIGN KEY (person_id) REFERENCES people(id),
        FOREIGN KEY (other_person_id) REFERENCES people(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS identities (
        person_id TEXT PRIMARY KEY,
        nombre TEXT,
        descripcion_corta TEXT,
        biografia_base TEXT,
        valores TEXT,
        estilo_habla TEXT,
        instrucciones_simulacion TEXT,
        edad_actual INTEGER,
        anio_nacimiento_aprox INTEGER,
        ciudad_origen TEXT,
        pais_origen TEXT,
        rasgos_clave TEXT,
        temas_biograficos TEXT,
        FOREIGN KEY (person_id) REFERENCES people(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS personas (
        id INTEGER PRIMARY KEY,
        nombre TEXT,
        descripcion TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lugares (
        id INTEGER PRIMARY KEY,
        nombre TEXT,
        descripcion TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS objetos (
        id INTEGER PRIMARY KEY,
        nombre TEXT,
        descripcion TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS emociones (
        id INTEGER PRIMARY KEY,
        nombre TEXT,
        descripcion TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS etiquetas (
        id INTEGER PRIMARY KEY,
        nombre TEXT,
        descripcion TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memorias (
        id INTEGER PRIMARY KEY,
        titulo TEXT,
        descripcion TEXT,
        fecha_inicio TEXT,
        fecha_fin TEXT,
        precision_temporal TEXT,
        nivel_certeza INTEGER,
        intensidad_emocional INTEGER,
        fuente TEXT,
        notas TEXT,
        person_id TEXT,
        FOREIGN KEY (person_id) REFERENCES people(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memoria_persona (
        memoria_id INTEGER,
        persona_id INTEGER,
        PRIMARY KEY (memoria_id, persona_id),
        FOREIGN KEY (memoria_id) REFERENCES memorias(id),
        FOREIGN KEY (persona_id) REFERENCES personas(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memoria_lugar (
        memoria_id INTEGER,
        lugar_id INTEGER,
        PRIMARY KEY (memoria_id, lugar_id),
        FOREIGN KEY (memoria_id) REFERENCES memorias(id),
        FOREIGN KEY (lugar_id) REFERENCES lugares(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memoria_objeto (
        memoria_id INTEGER,
        objeto_id INTEGER,
        PRIMARY KEY (memoria_id, objeto_id),
        FOREIGN KEY (memoria_id) REFERENCES memorias(id),
        FOREIGN KEY (objeto_id) REFERENCES objetos(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memoria_emocion (
        memoria_id INTEGER,
        emocion_id INTEGER,
        PRIMARY KEY (memoria_id, emocion_id),
        FOREIGN KEY (memoria_id) REFERENCES memorias(id),
        FOREIGN KEY (emocion_id) REFERENCES emociones(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memoria_etiqueta (
        memoria_id INTEGER,
        etiqueta_id INTEGER,
        PRIMARY KEY (memoria_id, etiqueta_id),
        FOREIGN KEY (memoria_id) REFERENCES memorias(id),
        FOREIGN KEY (etiqueta_id) REFERENCES etiquetas(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS simulacion_historial (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT,
        role TEXT,
        content TEXT,
        evidencias_json TEXT,
        FOREIGN KEY (person_id) REFERENCES people(id)
    )
    """)

    # Insert houses
    for house in census["houses"]:
        cursor.execute("""
        INSERT OR REPLACE INTO houses (id, residents_count)
        VALUES (?, ?)
        """, (house["id"], len(house["residents"])))

    # Insert people
    for person in census["people"]:
        cursor.execute("""
        INSERT OR REPLACE INTO people (
            id, full_name, first_name, surname_1, surname_2, gender, age, role,
            house_id, school_id, highschool_id, university_id, workplace_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            person["id"], full_name(person), person.get("first_name"), person.get("surname_1"),
            person.get("surname_2"), person.get("gender"), person.get("age"), person.get("role"),
            person.get("house_id"), person.get("school_id"), person.get("highschool_id"),
            person.get("university_id"), person.get("workplace_id")
        ))

        # Insert relations
        for relation in person.get("relations", []):
            cursor.execute("""
            INSERT OR REPLACE INTO relations (person_id, other_person_id, relation_type)
            VALUES (?, ?, ?)
            """, (person["id"], relation["person_id"], relation["type"]))

    # Insert autobiographical data
    for person_data in people_data:
        if "identity" in person_data:
            identity = person_data["identity"]
            cursor.execute("""
            INSERT OR REPLACE INTO identities (
                person_id, nombre, descripcion_corta, biografia_base, valores, estilo_habla,
                instrucciones_simulacion, edad_actual, anio_nacimiento_aprox, ciudad_origen,
                pais_origen, rasgos_clave, temas_biograficos
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                person_data["source_person"]["id"], identity["nombre"], identity["descripcion_corta"],
                identity["biografia_base"], identity["valores"], identity["estilo_habla"],
                identity["instrucciones_simulacion"], identity["edad_actual"], identity["anio_nacimiento_aprox"],
                identity["ciudad_origen"], identity["pais_origen"], json.dumps(identity["rasgos_clave"]),
                json.dumps(identity["temas_biograficos"])
            ))

        # Insert entities
        for entity_type in ["personas", "lugares", "objetos", "emociones", "etiquetas"]:
            if entity_type in person_data:
                for entity in person_data[entity_type]:
                    cursor.execute(f"""
                    INSERT OR IGNORE INTO {entity_type} (nombre, descripcion)
                    VALUES (?, ?)
                    """, (entity["nombre"], entity["descripcion"]))

        # Insert memories
        if "memorias" in person_data:
            for memory in person_data["memorias"]:
                cursor.execute("""
                INSERT INTO memorias (
                    titulo, descripcion, fecha_inicio, fecha_fin, precision_temporal,
                    nivel_certeza, intensidad_emocional, fuente, notas, person_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory["titulo"], memory["descripcion"], memory.get("fecha_inicio"),
                    memory.get("fecha_fin"), memory["precision_temporal"], memory["nivel_certeza"],
                    memory["intensidad_emocional"], memory["fuente"], memory["notas"],
                    person_data["source_person"]["id"]
                ))

                # Get the last inserted memory ID
                memory_id = cursor.lastrowid

                # Insert memory links
                for entity_type, link_table in [
                    ("personas", "memoria_persona"),
                    ("lugares", "memoria_lugar"),
                    ("objetos", "memoria_objeto"),
                    ("emociones", "memoria_emocion"),
                    ("etiquetas", "memoria_etiqueta")
                ]:
                    if entity_type in memory:
                        for entity_name in memory[entity_type]:
                            # Get entity ID
                            cursor.execute(f"SELECT id FROM {entity_type} WHERE nombre = ?", (entity_name,))
                            result = cursor.fetchone()
                            if result:
                                entity_id = result[0]
                                cursor.execute(f"""
                                INSERT INTO {link_table} (memoria_id, {entity_type[:-1]}_id)
                                VALUES (?, ?)
                                """, (memory_id, entity_id))

        # Insert simulation history
        if "simulacion_historial" in person_data:
            for history_item in person_data["simulacion_historial"]:
                cursor.execute("""
                INSERT INTO simulacion_historial (person_id, role, content, evidencias_json)
                VALUES (?, ?, ?, ?)
                """, (
                    person_data["source_person"]["id"], history_item["role"],
                    history_item["content"], history_item["evidencias_json"]
                ))

    conn.commit()
    conn.close()

# ===== Main Function =====
def main() -> None:
    parser = argparse.ArgumentParser(description="Genera una sociedad simulada y autobiografías para cada persona, guardando todo en una base de datos SQLite.")
    parser.add_argument("--db-path", default="society.db", help="Ruta a la base de datos SQLite de salida")
    parser.add_argument("--out-dir", default="society_personas_out", help="Carpeta base de salida para archivos intermedios")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo local de Ollama")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Endpoint /api/chat de Ollama")
    parser.add_argument("--seed-hint", default="", help="Ruta opcional a schema/seed de ejemplo. Si se omite, se intenta detectar.")
    parser.add_argument("--max-people", type=int, default=None, help="Limita cuántas personas procesar")
    parser.add_argument("--person-id", action="append", default=[], help="Procesa solo ciertos IDs, repetible")
    parser.add_argument("--target-personas", type=int, default=DEFAULT_TARGETS["personas"])
    parser.add_argument("--target-lugares", type=int, default=DEFAULT_TARGETS["lugares"])
    parser.add_argument("--target-objetos", type=int, default=DEFAULT_TARGETS["objetos"])
    parser.add_argument("--target-emociones", type=int, default=DEFAULT_TARGETS["emociones"])
    parser.add_argument("--target-etiquetas", type=int, default=DEFAULT_TARGETS["etiquetas"])
    parser.add_argument("--target-memorias", type=int, default=DEFAULT_TARGETS["memorias"])
    parser.add_argument("--target-historial", type=int, default=DEFAULT_TARGETS["historial"])
    parser.add_argument("--entity-passes", type=int, default=DEFAULT_BATCHES["entity_passes"])
    parser.add_argument("--memories-per-batch", type=int, default=DEFAULT_BATCHES["memories_per_batch"])
    parser.add_argument("--force-regenerate-history", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria opcional para la generación de la sociedad")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("[INFO] Generando censo de la sociedad...")
    census, warnings = generate_census()
    census_path = "census.json"
    with open(census_path, "w", encoding="utf-8") as f:
        json.dump(census, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Censo generado y guardado en {census_path}")

    idx = build_indexes(census)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OllamaClient(args.ollama_url, args.model)

    targets = {
        "personas": args.target_personas,
        "lugares": args.target_lugares,
        "objetos": args.target_objetos,
        "emociones": args.target_emociones,
        "etiquetas": args.target_etiquetas,
        "memorias": args.target_memorias,
        "historial": args.target_historial,
    }

    selected_people = select_people(idx["people"], args.person_id, args.max_people)
    if not selected_people:
        raise RuntimeError("No hay personas seleccionadas para procesar")

    manifest = {
        "created_at": now_iso(),
        "model": args.model,
        "census": census_path,
        "total_selected": len(selected_people),
        "people": [],
    }

    seed_hint_text = read_optional_text(args.seed_hint)
    people_data = []

    for i, person in enumerate(selected_people, start=1):
        name = full_name(person)
        print(f"\n=== [{i}/{len(selected_people)}] {name} ({person['id']}) ===")
        social_context = collect_social_context(person, idx)
        generate_for_person(client, person, social_context, out_dir, seed_hint_text, targets, args.entity_passes, args.memories_per_batch, args.force_regenerate_history)

        person_dir = build_person_dir(out_dir, person)
        state_path = person_dir / "state.json"
        with open(state_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
            people_data.append(state)

        manifest["people"].append({
            "id": person["id"],
            "nombre": name,
            "dir": str(person_dir),
            "sql": str(person_dir / "seed.sql"),
            "state": str(person_dir / "state.json"),
        })

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Creando base de datos SQLite en {args.db_path}...")
    create_database(args.db_path, census, people_data)
    print(f"[OK] Base de datos guardada: {args.db_path}")
    print(f"Manifest: {manifest_path}")
    print("[OK] Proceso completado.")

if __name__ == "__main__":
    main()
