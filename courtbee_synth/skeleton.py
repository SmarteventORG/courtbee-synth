import datetime
import random
from datetime import date, timedelta
import unicodedata
from faker import Faker

from courtbee_synth.models import Player, PlayerInvite, Sport

AGE_BANDS = [(5, 12), (13, 17), (18, 29), (30, 49), (50, 69), (70, 85)]
AGE_WEIGHTS = [8, 10, 22, 40, 17, 3]
MOBILE_PREFIXES = [
    "901", "903", "905", "908", "910", "911", "915",
    "917", "940", "944", "948", "949", "950", "951",
]
SKILL_LEVEL_WEIGHTS = [10, 14, 16, 16, 14, 11, 8, 6, 3, 2]


def max_skill_level_for_age(age: int) -> int:
    if age < 10:
        return 3
    if age < 13:
        return 5
    if age < 16:
        return 8
    return 10


def to_ascii_lower(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return "".join(ch for ch in ascii_text.lower() if ch.isalnum())


def make_player(fake: Faker) -> Player:
    player_id = fake.uuid4()
    if fake.random.random() < 0.5:
        first_name, last_name = fake.first_name_male(), fake.last_name_male()
    else:
        first_name, last_name = fake.first_name_female(), fake.last_name_female()
    name = f"{first_name} {last_name}"
    local_part = f"{to_ascii_lower(first_name)}.{to_ascii_lower(last_name)}"
    email = f"{local_part}.{player_id[:6]}@example.com"

    age_min, age_max = fake.random.choices(AGE_BANDS, weights=AGE_WEIGHTS)[0]
    age = fake.random.randint(age_min, age_max)

    tel_number = None
    if age >= 12 and fake.random.random() < 0.9:
        prefix = fake.random.choice(MOBILE_PREFIXES)
        digits = fake.random.randint(0, 999_999)
        tel_number = f"+421 {prefix} {digits // 1000:03d} {digits % 1000:03d}"

    skill_level = None
    if fake.random.random() < 0.8:
        max_level = max_skill_level_for_age(age)
        levels = range(1, max_level + 1)
        weights = SKILL_LEVEL_WEIGHTS[:max_level]
        skill_level = fake.random.choices(levels, weights=weights)[0]

    return Player(
        id=player_id,
        name=name,
        email=email,
        tel_number=tel_number,
        age=age,
        skill_level=skill_level,
        allow_play_invites=fake.boolean(),
    )

def make_invite(fake: Faker, player: Player, today: date) -> PlayerInvite:

    time_start_mins = fake.random.randrange(420, 1230, 30)
    reservation_length_mins = fake.random.randrange(60, 150, 30)
    time_end_mins = time_start_mins + reservation_length_mins

    time_start_hour, time_start_minute = divmod(time_start_mins, 60)
    time_end_hour, time_end_minute = divmod(time_end_mins, 60)

    if player.skill_level is None:
        level_min, level_max = 1, 10
    else:
        level_min = max(1, player.skill_level - fake.random.randint(0, 2))
        level_max = min(10, player.skill_level + fake.random.randint(0, 2))

    return PlayerInvite(
        id=fake.uuid4(),
        author_id=player.id,
        sport=fake.random_element(elements=(Sport.PADEL, Sport.PICKLEBALL, Sport.TENIS)),
        date=today + timedelta(days=fake.random.randint(1, 14)),
        time_from=datetime.time(hour=time_start_hour, minute=time_start_minute),
        time_to=datetime.time(hour=time_end_hour, minute=time_end_minute),
        level_min=level_min,
        level_max=level_max,
        num_players=fake.random_element(elements=(2, 4)),
        # text_sk - added by AI
    )

def main() -> None:
    seed = 42
    rng = random.Random(seed)
    fake = Faker("sk_SK")
    fake.seed_instance(seed)
    today = date(2026, 10, 1)

    for _ in range(5):
        player = make_player(fake)
        print(player.model_dump_json())
        if player.allow_play_invites:
            print("  ", make_invite(fake, player, today).model_dump_json())


if __name__ == "__main__":
    main()
