"""
Copyright (c) 2024 WindyLab of Westlake University, China
All rights reserved.

This software is provided "as is" without warranty of any kind, either
express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, or non-infringement.
In no event shall the authors or copyright holders be liable for any
claim, damages, or other liability, whether in an action of contract,
tort, or otherwise, arising from, out of, or in connection with the
software or the use or other dealings in the software.
"""


class QuadTree:
    MAX_ENTITIES = 4
    MAX_LEVELS = 5

    def __init__(self, x, y, width, height, level=0):
        self.bounds = {"x": x, "y": y, "width": width, "height": height}
        self.level = level
        self.entities = []
        self.nodes = []

    def insert(self, entity):
        if self.nodes:
            index = self.get_index(entity)
            if index != -1:
                self.nodes[index].insert(entity)
                return

        self.entities.append(entity)

        if len(self.entities) > self.MAX_ENTITIES and self.level < self.MAX_LEVELS:
            if not self.nodes:
                self.split()

            i = 0
            while i < len(self.entities):
                index = self.get_index(self.entities[i])
                if index != -1:
                    self.nodes[index].insert(self.entities.pop(i))
                else:
                    i += 1

    def split(self):
        sub_width = self.bounds["width"] / 2
        sub_height = self.bounds["height"] / 2
        x = self.bounds["x"]
        y = self.bounds["y"]

        self.nodes.append(QuadTree(x, y, sub_width, sub_height, self.level + 1))
        self.nodes.append(
            QuadTree(x + sub_width, y, sub_width, sub_height, self.level + 1)
        )
        self.nodes.append(
            QuadTree(x, y + sub_height, sub_width, sub_height, self.level + 1)
        )
        self.nodes.append(
            QuadTree(
                x + sub_width, y + sub_height, sub_width, sub_height, self.level + 1
            )
        )

    def get_index(self, entity):
        index = -1
        vertical_midpoint = self.bounds["x"] + (self.bounds["width"] / 2)
        horizontal_midpoint = self.bounds["y"] + (self.bounds["height"] / 2)

        top_quadrant = (
            entity.position[1] < horizontal_midpoint
            and entity.position[1] + entity.size / 2 < horizontal_midpoint
        )
        bottom_quadrant = entity.position[1] > horizontal_midpoint

        if (
            entity.position[0] < vertical_midpoint
            and entity.position[0] + entity.size / 2 < vertical_midpoint
        ):
            if top_quadrant:
                index = 0
            elif bottom_quadrant:
                index = 2
        elif entity.position[0] > vertical_midpoint:
            if top_quadrant:
                index = 1
            elif bottom_quadrant:
                index = 3

        return index

    def retrieve(self, entity):
        index = self.get_index(entity)
        return_entities = self.entities.copy()

        if self.nodes:
            if index != -1:
                return_entities.extend(self.nodes[index].retrieve(entity))
            else:
                for node in self.nodes:
                    return_entities.extend(node.retrieve(entity))

        return return_entities

    def clear(self):
        self.entities = []
        for node in self.nodes:
            node.clear()
        self.nodes = []

    def remove(self, entity):
        """
        Remove an entity from the QuadTree.
        """
        index = self.get_index(entity)
        if index != -1 and self.nodes:
            self.nodes[index].remove(entity)
        elif entity in self.entities:
            self.entities.remove(entity)

    def update(self, entity):
        """
        Update the position of an entity in the QuadTree.
        """
        self.remove(entity)
        self.insert(entity)
