from typing import Literal

from pydantic import BaseModel


class Nop(BaseModel): ...


class Chunk(BaseModel):
    tag: Literal["observation", "action"] | str
    text: str


class ItemCompleted(BaseModel): ...


class TurnCompleted(BaseModel): ...


Event = Chunk | ItemCompleted | Nop | TurnCompleted


class Prompt(BaseModel):
    text: str


class Steer(BaseModel):
    text: str


class Stop(BaseModel): ...


Interrupt = Nop | Steer | Stop | Prompt
