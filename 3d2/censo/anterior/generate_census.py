
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import random
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import networkx as nx

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


def weighted_house_size():
    # Más hogares medianos y grandes para que exista una base demográfica
    # más sostenible en generaciones futuras.
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
        # Las casas de 2 personas no tienen hijos bajo la regla pedida,
        # pero favorecemos algo más la pareja hombre-mujer para que haya
        # un mayor potencial reproductivo en la sociedad.
        pattern = random.choices(
            ["male_female_pair", "male_male_pair", "female_female_pair"],
            weights=[0.68, 0.16, 0.16],
            k=1,
        )[0]
    else:
        # Para tamaños 3-5 priorizamos hogares sostenibles con hijos,
        # aunque seguimos permitiendo combinaciones sin hijos.
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

        # grandparent relation if group has elder + adult child generation
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

    # limited same-house sibling inference only when surnames match
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

    # Children inherit from parents when both exist
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

    # Siblings with shared parents must share surnames
    for person in people:
        sibling_ids = [r["person_id"] for r in person["relations"] if r["type"] == "sibling"]
        for sid in sibling_ids:
            sib = people_by_id[sid]
            my_parents = {r["person_id"] for r in person["relations"] if r["type"] == "parent"}
            sib_parents = {r["person_id"] for r in sib["relations"] if r["type"] == "parent"}
            if my_parents and my_parents == sib_parents:
                sib["surname_1"] = person["surname_1"]
                sib["surname_2"] = person["surname_2"]

    # Ensure everybody has both surnames
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


def is_family_type(rel_type):
    return rel_type in FAMILY_EDGE_TYPES


def is_weak_type(rel_type):
    return rel_type in WEAK_EDGE_TYPES


def build_graph(census):
    G = nx.Graph()
    people = census.get("people", [])
    people_by_id = {p["id"]: p for p in people}

    for p in people:
        pid = p["id"]
        G.add_node(
            pid,
            label=full_name(p),
            house_id=p.get("house_id", ""),
            age=p.get("age", 0),
            gender=p.get("gender", ""),
            role=p.get("role", ""),
        )

    seen = set()

    for p in people:
        src = p["id"]
        for rel in p.get("relations", []):
            dst = rel.get("person_id")
            rel_type = rel.get("type", "unknown")

            if not dst or dst not in people_by_id or dst == src:
                continue

            key = tuple(sorted((src, dst))) + (rel_type,)
            if key in seen:
                continue
            seen.add(key)

            if G.has_edge(src, dst):
                G[src][dst]["types"].add(rel_type)
            else:
                G.add_edge(src, dst, types={rel_type})

    return G


def count_relation_types(census):
    counts = Counter()
    seen = set()

    for p in census["people"]:
        for r in p["relations"]:
            a = p["id"]
            b = r["person_id"]
            t = r["type"]
            key = tuple(sorted((a, b))) + (t,)
            if key in seen:
                continue
            seen.add(key)
            counts[t] += 1

    return counts


def build_family_graph(G):
    FG = nx.Graph()
    FG.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        if any(is_family_type(t) for t in d["types"]):
            FG.add_edge(u, v, **d)
    return FG


def house_members(G):
    houses = defaultdict(list)
    for n, attrs in G.nodes(data=True):
        houses[attrs.get("house_id", "unknown_house")].append(n)
    return dict(houses)


def split_house_into_family_components(FG, members):
    sub = FG.subgraph(members).copy()
    comps = [list(c) for c in nx.connected_components(sub)]

    covered = set()
    for c in comps:
        covered.update(c)

    leftovers = [m for m in members if m not in covered]
    for m in leftovers:
        comps.append([m])

    comps.sort(key=len, reverse=True)
    return comps


def build_house_graph(G):
    HG = nx.Graph()

    houses = house_members(G)
    for house_id, members in houses.items():
        HG.add_node(house_id, size=len(members))

    edge_weights = defaultdict(float)

    for u, v, d in G.edges(data=True):
        hu = G.nodes[u].get("house_id", "unknown_house")
        hv = G.nodes[v].get("house_id", "unknown_house")
        if hu == hv:
            continue

        weak_count = sum(1 for t in d["types"] if is_weak_type(t))
        if weak_count <= 0:
            continue

        key = tuple(sorted((hu, hv)))
        edge_weights[key] += weak_count

    for (hu, hv), w in edge_weights.items():
        HG.add_edge(hu, hv, weight=w)

    return HG


def compute_house_centers_by_weak_ties(G, seed=HOUSE_LAYOUT_SEED):
    HG = build_house_graph(G)

    if HG.number_of_nodes() == 0:
        return {}

    if HG.number_of_nodes() == 1:
        only = next(iter(HG.nodes()))
        return {only: (0.0, 0.0)}

    pos = nx.spring_layout(
        HG,
        seed=seed,
        k=HOUSE_LAYOUT_K,
        iterations=HOUSE_LAYOUT_ITERATIONS,
        weight="weight",
        scale=40.0,
    )

    return {k: (float(v[0]), float(v[1])) for k, v in pos.items()}


def polar_to_xy(r, a):
    return (r * math.cos(a), r * math.sin(a))


def add_vec(a, b):
    return (a[0] + b[0], a[1] + b[1])


def choose_family_anchor_offsets(num_components, seed):
    rng = random.Random(seed)

    if num_components <= 0:
        return []

    if num_components == 1:
        return [(0.0, 0.0)]

    offsets = []
    local_radius = FAMILY_CLUSTER_BASE_RADIUS + FAMILY_CLUSTER_RADIUS_STEP * num_components

    for i in range(num_components):
        ang = (2.0 * math.pi * i / num_components) + rng.uniform(-0.10, 0.10)
        rr = local_radius + rng.uniform(-0.35, 0.35)
        offsets.append(polar_to_xy(rr, ang))

    return offsets


def layout_small_family_star(G, members, anchor, seed):
    rng = random.Random(seed)
    pos = {}

    if not members:
        return pos

    attrs = {m: G.nodes[m] for m in members}

    family_degree = {}
    for m in members:
        deg = 0
        for nb in G.neighbors(m):
            if nb in members:
                edge_types = G[m][nb]["types"]
                if any(is_family_type(t) for t in edge_types):
                    deg += 1
        family_degree[m] = deg

    spouses = []
    for m in members:
        for nb in G.neighbors(m):
            if nb in members and ("spouse" in G[m][nb]["types"] or "partner" in G[m][nb]["types"]):
                spouses.append(m)
                break
    spouses = list(dict.fromkeys(spouses))

    adults = [m for m in members if attrs[m].get("age", 0) >= 18]

    central = []
    for m in spouses:
        if m not in central:
            central.append(m)

    adults_sorted = sorted(
        adults,
        key=lambda x: (family_degree[x], attrs[x].get("age", 0)),
        reverse=True
    )
    for m in adults_sorted:
        if len(central) >= 2:
            break
        if m not in central:
            central.append(m)

    if not central:
        central = [max(members, key=lambda x: (family_degree[x], attrs[x].get("age", 0)))]

    if len(central) == 1:
        pos[central[0]] = anchor
    elif len(central) == 2:
        pos[central[0]] = add_vec(anchor, (-0.95, 0.0))
        pos[central[1]] = add_vec(anchor, (0.95, 0.0))
    else:
        for i, m in enumerate(central):
            a = 2.0 * math.pi * i / len(central)
            pos[m] = add_vec(anchor, polar_to_xy(1.05, a))

    remaining = [m for m in members if m not in pos]

    if not remaining:
        return pos

    ring1 = []
    ring2 = []

    for m in remaining:
        age = attrs[m].get("age", 0)
        if age < 18:
            ring1.append(m)
        else:
            ring2.append(m)

    if ring1:
        r1 = 2.2 + 0.30 * min(6, len(ring1))
        angle0 = rng.uniform(0, 2.0 * math.pi)
        for i, m in enumerate(ring1):
            a = angle0 + 2.0 * math.pi * i / len(ring1) + rng.uniform(-0.08, 0.08)
            pos[m] = add_vec(anchor, polar_to_xy(r1, a))

    if ring2:
        r2 = 3.4 + 0.25 * min(8, len(ring2))
        angle0 = rng.uniform(0, 2.0 * math.pi)
        for i, m in enumerate(ring2):
            a = angle0 + 2.0 * math.pi * i / len(ring2) + rng.uniform(-0.08, 0.08)
            pos[m] = add_vec(anchor, polar_to_xy(r2, a))

    return pos


def build_positions(G, seed=1234):
    FG = build_family_graph(G)
    houses = house_members(G)
    house_ids = sorted(houses.keys())

    house_centers = compute_house_centers_by_weak_ties(G, seed=seed)

    for i, house_id in enumerate(house_ids):
        if house_id not in house_centers:
            a = 2.0 * math.pi * i / max(1, len(house_ids))
            house_centers[house_id] = polar_to_xy(30.0, a)

    pos = {}

    for hi, house_id in enumerate(house_ids):
        members = houses[house_id]
        house_center = house_centers[house_id]

        family_components = split_house_into_family_components(FG, members)
        anchor_offsets = choose_family_anchor_offsets(len(family_components), seed + hi * 17)

        for ci, comp in enumerate(family_components):
            anchor = add_vec(house_center, anchor_offsets[ci])
            local_pos = layout_small_family_star(G, comp, anchor, seed + hi * 100 + ci * 11)

            for node, p in local_pos.items():
                pos[node] = p

    pos = resolve_house_rectangle_overlaps(G, pos, iterations=320, margin=1.9, padding=1.2, step_scale=0.72)
    return pos



def etiquetar_casa(house_id):
    return str(house_id).replace("house_", "casa_")


def agrupar_nodos_por_casa(G, pos, nodes=None):
    houses = defaultdict(list)
    node_iter = nodes if nodes is not None else G.nodes()
    for n in node_iter:
        if n not in pos or n not in G.nodes:
            continue
        house_id = G.nodes[n].get("house_id", "unknown_house")
        houses[house_id].append(n)
    return dict(houses)


def compute_house_bboxes(G, pos, nodes=None, margin=1.8):
    boxes = {}
    grouped = agrupar_nodos_por_casa(G, pos, nodes=nodes)
    for house_id, members in grouped.items():
        if not members:
            continue
        xs = [pos[n][0] for n in members]
        ys = [pos[n][1] for n in members]
        boxes[house_id] = {
            "min_x": min(xs) - margin,
            "max_x": max(xs) + margin,
            "min_y": min(ys) - margin,
            "max_y": max(ys) + margin,
            "members": members[:],
        }
    return boxes


def boxes_overlap(a, b, padding=0.8):
    return not (
        a["max_x"] + padding <= b["min_x"] or
        b["max_x"] + padding <= a["min_x"] or
        a["max_y"] + padding <= b["min_y"] or
        b["max_y"] + padding <= a["min_y"]
    )


def shift_house_nodes(pos, members, dx, dy):
    for node in members:
        x, y = pos[node]
        pos[node] = (x + dx, y + dy)


def resolve_house_rectangle_overlaps(G, pos, iterations=300, margin=1.8, padding=1.0, step_scale=0.55):
    if not pos:
        return pos

    for _ in range(iterations):
        boxes = compute_house_bboxes(G, pos, margin=margin)
        house_ids = sorted(boxes.keys())
        moved_any = False

        for i in range(len(house_ids)):
            for j in range(i + 1, len(house_ids)):
                a_id = house_ids[i]
                b_id = house_ids[j]
                a = boxes[a_id]
                b = boxes[b_id]

                if not boxes_overlap(a, b, padding=padding):
                    continue

                ax = (a["min_x"] + a["max_x"]) * 0.5
                ay = (a["min_y"] + a["max_y"]) * 0.5
                bx = (b["min_x"] + b["max_x"]) * 0.5
                by = (b["min_y"] + b["max_y"]) * 0.5

                dx = bx - ax
                dy = by - ay

                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    angle = (i * 92821 + j * 68917) % 360
                    angle = math.radians(angle)
                    dx = math.cos(angle)
                    dy = math.sin(angle)

                overlap_x = min(a["max_x"], b["max_x"]) - max(a["min_x"], b["min_x"]) + padding
                overlap_y = min(a["max_y"], b["max_y"]) - max(a["min_y"], b["min_y"]) + padding

                if overlap_x <= 0 or overlap_y <= 0:
                    continue

                if abs(dx) >= abs(dy):
                    move = overlap_x * 0.5 * step_scale
                    sx = 1.0 if dx >= 0 else -1.0
                    shift_house_nodes(pos, a["members"], -sx * move, 0.0)
                    shift_house_nodes(pos, b["members"], sx * move, 0.0)
                else:
                    move = overlap_y * 0.5 * step_scale
                    sy = 1.0 if dy >= 0 else -1.0
                    shift_house_nodes(pos, a["members"], 0.0, -sy * move)
                    shift_house_nodes(pos, b["members"], 0.0, sy * move)

                moved_any = True

        if not moved_any:
            break

    return pos


def draw_house_rectangles(ax, G, pos, nodes=None, margin=1.8):
    boxes = compute_house_bboxes(G, pos, nodes=nodes, margin=margin)
    cmap = cm.get_cmap("tab20", max(1, len(boxes)))

    for i, house_id in enumerate(sorted(boxes.keys())):
        box = boxes[house_id]
        min_x = box["min_x"]
        max_x = box["max_x"]
        min_y = box["min_y"]
        max_y = box["max_y"]

        rect = patches.Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            linewidth=1.8,
            edgecolor=cmap(i),
            facecolor=cmap(i),
            alpha=0.08,
            zorder=0,
        )
        ax.add_patch(rect)
        ax.text(
            min_x,
            max_y + 0.15,
            etiquetar_casa(house_id),
            fontsize=7,
            color=cmap(i),
            verticalalignment="bottom",
            horizontalalignment="left",
            zorder=1,
        )




def node_color(attrs):
    gender = attrs.get("gender", "")
    if gender == "male":
        return "#9ecae1"
    if gender == "female":
        return "#fcbfd2"
    return "#dddddd"


def node_size(age):
    if age <= 12:
        return 180
    if age <= 17:
        return 210
    if age <= 22:
        return 235
    if age <= 65:
        return 270
    return 300


def add_legend(ax):
    x0 = 0.02
    y0 = 0.98
    dy = 0.028

    items = [
        ("#d62728", "pareja"),
        ("#004c99", "padre / madre / hijo"),
        ("#1f9d55", "hermano / hermana"),
        ("#999999", "amistad"),
    ]

    for i, (color, text) in enumerate(items):
        y = y0 - i * dy
        ax.plot([x0, x0 + 0.03], [y, y], transform=ax.transAxes, color=color,
                linewidth=3.0 if i < 3 else 1.0, alpha=0.95 if i < 3 else 0.5)
        ax.text(x0 + 0.035, y, text, transform=ax.transAxes, fontsize=10,
                va="center", ha="left", color="#222222")


def draw_society_chart(G, output_path, seed=1234):
    pos = build_positions(G, seed=seed)

    plt.figure(figsize=(24, 24))
    ax = plt.gca()
    ax.set_facecolor("white")

    weak_edges = []
    parent_child_edges = []
    sibling_edges = []
    spouse_edges = []

    for u, v, d in G.edges(data=True):
        types = d["types"]
        if "friend" in types:
            weak_edges.append((u, v))
        elif "parent" in types or "child" in types:
            parent_child_edges.append((u, v))
        elif "sibling" in types:
            sibling_edges.append((u, v))
        elif "spouse" in types or "partner" in types:
            spouse_edges.append((u, v))

    draw_house_rectangles(ax, G, pos)

    nx.draw_networkx_edges(G, pos, edgelist=weak_edges, edge_color="#999999", width=0.55, alpha=0.14, style="solid")
    nx.draw_networkx_edges(G, pos, edgelist=parent_child_edges, edge_color="#004c99", width=3.8, alpha=0.95, connectionstyle="arc3,rad=0.10")
    nx.draw_networkx_edges(G, pos, edgelist=sibling_edges, edge_color="#1f9d55", width=2.8, alpha=0.95, connectionstyle="arc3,rad=0.04")
    nx.draw_networkx_edges(G, pos, edgelist=spouse_edges, edge_color="#d62728", width=4.2, alpha=0.95, connectionstyle="arc3,rad=0.0")

    node_sizes = [node_size(G.nodes[n].get("age", 0)) for n in G.nodes()]
    node_colors = [node_color(G.nodes[n]) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors="#222222", linewidths=0.8)

    labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=LABEL_FONT_SIZE, font_family="sans-serif", verticalalignment="bottom")

    add_legend(ax)
    plt.title("Mapa de relaciones de la minisociedad", fontsize=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close()


def draw_family_chart(G, output_path, seed=1234):
    pos = build_positions(G, seed=seed)

    plt.figure(figsize=(24, 24))
    ax = plt.gca()
    ax.set_facecolor("white")

    parent_child_edges = []
    sibling_edges = []
    spouse_edges = []
    family_nodes = set()

    for u, v, d in G.edges(data=True):
        types = d["types"]
        if "parent" in types or "child" in types:
            parent_child_edges.append((u, v))
            family_nodes.update([u, v])
        elif "sibling" in types:
            sibling_edges.append((u, v))
            family_nodes.update([u, v])
        elif "spouse" in types or "partner" in types:
            spouse_edges.append((u, v))
            family_nodes.update([u, v])

    H = G.subgraph(family_nodes).copy()

    draw_house_rectangles(ax, G, pos, nodes=H.nodes())

    nx.draw_networkx_edges(H, pos, edgelist=parent_child_edges, edge_color="#004c99", width=4.0, alpha=0.98, connectionstyle="arc3,rad=0.12")
    nx.draw_networkx_edges(H, pos, edgelist=sibling_edges, edge_color="#1f9d55", width=3.0, alpha=0.98, connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edges(H, pos, edgelist=spouse_edges, edge_color="#d62728", width=4.3, alpha=0.98, connectionstyle="arc3,rad=0.0")

    node_sizes = [node_size(H.nodes[n].get("age", 0)) for n in H.nodes()]
    node_colors = [node_color(H.nodes[n]) for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color=node_colors, edgecolors="#222222", linewidths=0.8)

    labels = {n: H.nodes[n].get("label", n) for n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=LABEL_FONT_SIZE, font_family="sans-serif", verticalalignment="bottom")

    add_legend(ax)
    plt.title("Mapa de relaciones familiares", fontsize=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close()


def draw_parent_child_chart(G, output_path, seed=1234):
    pos = build_positions(G, seed=seed)

    plt.figure(figsize=(24, 24))
    ax = plt.gca()
    ax.set_facecolor("white")

    parent_child_edges = []
    nodes_used = set()

    for u, v, d in G.edges(data=True):
        types = d["types"]
        if "parent" in types or "child" in types:
            parent_child_edges.append((u, v))
            nodes_used.update([u, v])

    H = G.subgraph(nodes_used).copy()

    draw_house_rectangles(ax, G, pos, nodes=H.nodes())

    nx.draw_networkx_edges(H, pos, edgelist=parent_child_edges, edge_color="#004c99", width=4.5, alpha=0.98, connectionstyle="arc3,rad=0.14")

    node_sizes = [node_size(H.nodes[n].get("age", 0)) for n in H.nodes()]
    node_colors = [node_color(H.nodes[n]) for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color=node_colors, edgecolors="#222222", linewidths=0.8)

    labels = {n: H.nodes[n].get("label", n) for n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=LABEL_FONT_SIZE, font_family="sans-serif", verticalalignment="bottom")

    ax.plot([0.02, 0.05], [0.98, 0.98], transform=ax.transAxes, color="#004c99", linewidth=4.0)
    ax.text(0.055, 0.98, "padre / madre / hijo", transform=ax.transAxes, fontsize=10, va="center", ha="left", color="#222222")

    plt.title("Mapa de relaciones paterno-filiales", fontsize=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria opcional")
    parser.add_argument("--json-output", default="census.json", help="JSON de censo de salida")
    parser.add_argument("--society-png", default="society_chart.png", help="PNG de salida del mapa social completo")
    parser.add_argument("--family-png", default="family_chart.png", help="PNG de salida del mapa familiar")
    parser.add_argument("--parent-child-png", default="parent_child_chart.png", help="PNG de salida del mapa paterno-filial")
    parser.add_argument("--chart-seed", type=int, default=1234, help="Semilla de distribución para el dibujo")
    parser.add_argument("--warnings-output", default="validation_warnings.txt", help="Archivo de texto con advertencias de validación")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    census, warnings = generate_census()

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(census, f, ensure_ascii=False, indent=2)

    G = build_graph(census)
    draw_society_chart(G, args.society_png, seed=args.chart_seed)
    draw_family_chart(G, args.family_png, seed=args.chart_seed)
    draw_parent_child_chart(G, args.parent_child_png, seed=args.chart_seed)

    with open(args.warnings_output, "w", encoding="utf-8") as f:
        if warnings:
            for w in warnings:
                f.write(w + "\n")
        else:
            f.write("No hay advertencias de validación.\n")

    rel_counts = count_relation_types(census)

    print(f"JSON del censo generado:         {args.json_output}")
    print(f"Mapa social generado:            {args.society_png}")
    print(f"Mapa familiar generado:          {args.family_png}")
    print(f"Mapa paterno-filial generado:    {args.parent_child_png}")
    print(f"Archivo de advertencias:         {args.warnings_output}")
    print(f"Personas: {census['metadata']['generated_people']}")
    print(f"Casas: {census['metadata']['houses']}")
    print(f"Nodos del grafo: {G.number_of_nodes()}")
    print(f"Aristas del grafo: {G.number_of_edges()}")
    print(f"Errores de validación: {census['metadata']['validation_errors']}")
    print(f"Advertencias de validación: {census['metadata']['validation_warnings']}")
    print("Recuento de relaciones:")
    for k, v in sorted(rel_counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
