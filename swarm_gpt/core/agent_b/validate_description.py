# primitive_validator.py

import re
from difflib import SequenceMatcher
from swarm_gpt.core.motion_primitives import motion_primitives


class ValidateNameDesc:
    DOMAIN_TERMS = {"drone", "swarm", "primitive", "motion"}
    NAME_PATTERN = re.compile(r'[a-z][a-z0-9_]*')

    def __init__(
        self,
        existing_names: list[str] = list(motion_primitives.keys()),
        similarity_threshold: float = 0.85,
        desc_min_words: int = 10,
        desc_max_words: int = 60,
    ):
        self.existing_names = existing_names
        self.similarity_threshold = similarity_threshold
        self.desc_min_words = desc_min_words
        self.desc_max_words = desc_max_words


    def _valid_snake_case(self, name: str) -> bool:
        return bool(self.NAME_PATTERN.fullmatch(name))

    def _is_duplicate(self, name: str) -> bool:
        return name in self.existing_names

    def _is_too_similar(self, name: str) -> bool:
        return any(
            SequenceMatcher(None, name, existing).ratio() > self.similarity_threshold
            for existing in self.existing_names
        )

    def _valid_description(self, desc: str) -> bool:
        word_count = len(desc.split())
        has_domain_term = any(t in desc.lower() for t in self.DOMAIN_TERMS)
        return self.desc_min_words <= word_count <= self.desc_max_words and has_domain_term

    # def _name_in_description(self, name: str, desc: str) -> bool:
    #     tokens = [t for t in name.split("_") if len(t) > 3]
    #     if not tokens:
    #         return True  # name troppo corto per il check --> non penalizzare
    #     return any(t in desc.lower() for t in tokens)

    def validate(self, name: str, desc: str) -> tuple[bool, list[str]]:
        # name = result.get("name", "")
        # desc = result.get("description", "")
        print(f"name: {name}")
        print(f"desc: {desc}")
        errors = []

        if not self._valid_snake_case(name):
            errors.append(f"Invalid snake_case: '{name}'")

        if self._is_duplicate(name):
            errors.append(f"Duplicate name: '{name}'")
        elif self._is_too_similar(name):  # elif: inutile se già duplicato esatto
            errors.append(f"Too similar to existing name: '{name}'")

        if not self._valid_description(desc):
            errors.append(
                f"Description must be {self.desc_min_words}–{self.desc_max_words} words "
                f"and contain a domain term {self.DOMAIN_TERMS}"
            )

        # if not self._name_in_description(name, desc):
        #     errors.append(f"Name '{name}' semantically disconnected from description")

        return len(errors) == 0, errors