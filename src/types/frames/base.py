#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from dataclasses import dataclass, field

from src.common.utils.obj import obj_count, obj_id


@dataclass
class Frame:
    id: int = field(init=False)
    name: str = field(init=False)

    def __post_init__(self):
        self.id: int = obj_id()
        self.name: str = f"{self.__class__.__name__}#{obj_count(self)}"

    def __str__(self):
        return self.name
