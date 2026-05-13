from typing import Literal

from pydantic import BaseModel, Field


class Nop(BaseModel): ...


class Chunk(BaseModel):
    tag: Literal["observation", "action"] | str
    text: str


class Log(BaseModel):
    kind: str
    text: str = ""
    raw: dict[str, object] = Field(default_factory=dict)


class ItemCompleted(BaseModel): ...


class TurnCompleted(BaseModel): ...


Event = Chunk | Log | ItemCompleted | Nop | TurnCompleted


class Prompt(BaseModel):
    text: str


class Steer(BaseModel):
    text: str


class Stop(BaseModel): ...


Interrupt = Nop | Steer | Stop | Prompt
