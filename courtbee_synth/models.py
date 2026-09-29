from datetime import date, time
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Sport(StrEnum):
    PADEL = "padel"
    PICKLEBALL = "pickleball"
    TENIS = "tenis"


class PlayerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=255, pattern=r"^[^@\s]+@example\.com$")
    tel_number: str | None = Field(default=None, max_length=20)
    age: int | None = Field(default=None, ge=5, le=99)
    skill_level: int | None = Field(default=None, ge=1, le=10)
    allow_play_invites: bool = True


class Player(PlayerDraft):
    id: UUID


class PlayerInviteDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_times(self) -> Self:
        if self.time_from >= self.time_to:
            raise ValueError("time_to must not be before time_from")
        return self

    @model_validator(mode="after")
    def check_levels(self) -> Self:
        if self.level_min > self.level_max:
            raise ValueError("level_min must not be greater than level_max")
        return self

    sport: Sport
    date: date
    time_from: time
    time_to: time
    level_min: int = Field(ge=1, le=10)
    level_max: int = Field(ge=1, le=10)
    num_players: Literal[2, 4] = Field(default=2)
    text_sk: str | None = Field(default=None, max_length=255)


class PlayerInvite(PlayerInviteDraft):
    id: UUID
    author_id: UUID
