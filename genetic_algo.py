import random
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Константи для розкладу
DAYS_PER_WEEK = 5  # Кількість днів у тижні (без врахування субот)
LESSONS_PER_DAY = 4  # Кількість академічних годин на день
WEEK_TYPE = ['EVEN', 'ODD']  # Типи тижнів: парний та непарний

# Часові слоти з урахуванням парних/непарних тижнів
TIMESLOTS = [
    f"{week} - day {day + 1}, lesson {slot + 1}"
    for week in WEEK_TYPE
    for day in range(DAYS_PER_WEEK)
    for slot in range(LESSONS_PER_DAY)
]

@dataclass
class Event:
    timeslot: str
    group_ids: List[str]
    subject_id: str
    subject_name: str
    lecturer_id: str
    auditorium_id: str
    event_type: str
    subgroup_ids: Optional[Dict[str, str]] = None
    week_type: str = 'Both'

@dataclass
class Schedule:
    events: List[Event] = field(default_factory=list)
    hard_constraints_violations: int = 0
    soft_constraints_score: int = 0

    def add_event(self, event: Event) -> None:
        if event:
            self.events.append(event)

    def fitness(self, groups: Dict[str, Dict], lecturers: Dict[str, Dict], auditoriums: Dict[str, int]) -> int:
        self.hard_constraints_violations = 0
        self.soft_constraints_score = 0

        lecturer_times: Dict[Tuple[str, str], Event] = {}
        group_times: Dict[Tuple[str, str], Event] = {}
        subgroup_times: Dict[Tuple[str, str, str], Event] = {}
        auditorium_times: Dict[Tuple[str, str], Event] = {}
        lecturer_hours: Dict[Tuple[str, str], float] = {}

        # Додаткові структури для перевірки кафедральної узгодженості
        department_timeslots: Dict[str, Dict[str, List[Event]]] = {}
        # Формуємо структуру: {department_prefix: {timeslot: [events]}}
        for event in self.events:
            # Визначаємо департамент через префікс групи (припускаємо, що всі групи в події з одного департаменту)
            if event.group_ids:
                department_prefix = event.group_ids[0].split('-')[0]
            else:
                continue  # Якщо немає груп, пропускаємо подію

            if department_prefix not in department_timeslots:
                department_timeslots[department_prefix] = {}
            if event.timeslot not in department_timeslots[department_prefix]:
                department_timeslots[department_prefix][event.timeslot] = []
            department_timeslots[department_prefix][event.timeslot].append(event)

        for event in self.events:
            # Жорсткі обмеження
            self._check_hard_constraints(event, lecturer_times, group_times, subgroup_times,
                                         auditorium_times, lecturer_hours, lecturers)

            # М'які обмеження
            self._check_soft_constraints(event, groups, lecturers, auditoriums)

        # Додаткові жорсткі обмеження: узгодженість кафедри
        for department_prefix, timeslots in department_timeslots.items():
            department_groups = [gid for gid in groups if gid.startswith(department_prefix)]
            for timeslot, events in timeslots.items():
                # Визначаємо, які групи мають лекції в цьому часовому слоті
                groups_with_lecture = set()
                for event in events:
                    groups_with_lecture.update(event.group_ids)
                # Якщо деякі групи мають лекцію, а інші ні, це порушення
                if groups_with_lecture and groups_with_lecture != set(department_groups):
                    self.hard_constraints_violations += 1

        # Загальна оцінка
        total_score = self.hard_constraints_violations * 1000 + self.soft_constraints_score
        return total_score

    def _check_hard_constraints(
        self,
        event: Event,
        lecturer_times: Dict[Tuple[str, str], Event],
        group_times: Dict[Tuple[str, str], Event],
        subgroup_times: Dict[Tuple[str, str, str], Event],
        auditorium_times: Dict[Tuple[str, str], Event],
        lecturer_hours: Dict[Tuple[str, str], float],
        lecturers: Dict[str, Dict]
    ) -> None:
        # Перевірка зайнятості викладача
        lt_key = (event.lecturer_id, event.timeslot)
        if lt_key in lecturer_times:
            self.hard_constraints_violations += 1
        else:
            lecturer_times[lt_key] = event

        # Перевірка зайнятості груп та підгруп
        for group_id in event.group_ids:
            gt_key = (group_id, event.timeslot)
            if gt_key in group_times:
                self.hard_constraints_violations += 1
            else:
                group_times[gt_key] = event

            if event.subgroup_ids and group_id in event.subgroup_ids:
                subgroup_id = event.subgroup_ids[group_id]
                sgt_key = (group_id, subgroup_id, event.timeslot)
                if sgt_key in subgroup_times:
                    self.hard_constraints_violations += 1
                else:
                    subgroup_times[sgt_key] = event

        # Перевірка зайнятості аудиторії
        at_key = (event.auditorium_id, event.timeslot)
        if at_key in auditorium_times:
            existing_event = auditorium_times[at_key]
            if not (event.event_type == 'Лекція' and
                    existing_event.event_type == 'Лекція' and
                    event.lecturer_id == existing_event.lecturer_id):
                self.hard_constraints_violations += 1
        else:
            auditorium_times[at_key] = event

        # Перевірка навантаження викладача
        week = event.timeslot.split(' - ')[0]
        lecturer_hours_key = (event.lecturer_id, week)
        lecturer_hours[lecturer_hours_key] = lecturer_hours.get(lecturer_hours_key, 0) + 1.5
        if lecturer_hours[lecturer_hours_key] > lecturers[event.lecturer_id]['MaxHoursPerWeek']:
            self.hard_constraints_violations += 1

    def _check_soft_constraints(
        self,
        event: Event,
        groups: Dict[str, Dict],
        lecturers: Dict[str, Dict],
        auditoriums: Dict[str, int]
    ) -> None:
        total_group_size = sum(
            groups[g]['NumStudents'] // 2 if event.subgroup_ids and g in event.subgroup_ids else groups[g]['NumStudents']
            for g in event.group_ids
        )
        if auditoriums[event.auditorium_id] < total_group_size:
            self.soft_constraints_score += 1

        if event.subject_id not in lecturers[event.lecturer_id]['SubjectsCanTeach']:
            self.soft_constraints_score += 1

        if event.event_type not in lecturers[event.lecturer_id]['TypesCanTeach']:
            self.soft_constraints_score += 1

def generate_initial_population(
    pop_size: int,
    groups: Dict[str, Dict],
    subjects: List[Dict],
    lecturers: Dict[str, Dict],
    auditoriums: Dict[str, int]
) -> List[Schedule]:
    population: List[Schedule] = []
    for _ in range(pop_size):
        lecturer_times: Dict[Tuple[str, str], Event] = {}
        group_times: Dict[Tuple[str, str], Event] = {}
        subgroup_times: Dict[Tuple[str, str, str], Event] = {}
        auditorium_times: Dict[Tuple[str, str], Event] = {}
        schedule = Schedule()

        for subj in subjects:
            weeks = [subj['weekType']] if subj['weekType'] in WEEK_TYPE else WEEK_TYPE
            for week in weeks:
                # Додаємо лекції
                for _ in range(subj['numLectures']):
                    event = create_random_event(
                        subj, groups, lecturers, auditoriums, 'Лекція', week,
                        lecturer_times, group_times, subgroup_times, auditorium_times
                    )
                    if event:
                        schedule.add_event(event)

                # Додаємо практичні/лабораторні заняття
                for _ in range(subj['numPracticals']):
                    if subj['requiresSubgroups']:
                        for subgroup_id in groups[subj['groupID']]['Subgroups']:
                            subgroup_ids = {subj['groupID']: subgroup_id}
                            event = create_random_event(
                                subj, groups, lecturers, auditoriums, 'Практика', week,
                                lecturer_times, group_times, subgroup_times, auditorium_times, subgroup_ids
                            )
                            if event:
                                schedule.add_event(event)
                    else:
                        event = create_random_event(
                            subj, groups, lecturers, auditoriums, 'Практика', week,
                            lecturer_times, group_times, subgroup_times, auditorium_times
                        )
                        if event:
                            schedule.add_event(event)

        population.append(schedule)

    return population

def create_random_event(
    subj: Dict,
    groups: Dict[str, Dict],
    lecturers: Dict[str, Dict],
    auditoriums: Dict[str, int],
    event_type: str,
    week_type: str,
    lecturer_times: Dict[Tuple[str, str], Event],
    group_times: Dict[Tuple[str, str], Event],
    subgroup_times: Dict[Tuple[str, str, str], Event],
    auditorium_times: Dict[Tuple[str, str], Event],
    subgroup_ids: Optional[Dict[str, str]] = None
) -> Optional[Event]:
    available_timeslots = [t for t in TIMESLOTS if t.startswith(week_type)]
    if not available_timeslots:
        return None
    timeslot = random.choice(available_timeslots)

    # Знаходимо викладачів, які можуть викладати цей предмет і тип заняття
    suitable_lecturers = [
        lid for lid, l in lecturers.items()
        if subj['id'] in l['SubjectsCanTeach'] and event_type in l['TypesCanTeach']
    ]
    if not suitable_lecturers:
        return None

    # Вибираємо випадкового викладача, який не зайнятий у цей часовий слот
    random.shuffle(suitable_lecturers)
    lecturer_id = next((lid for lid in suitable_lecturers if (lid, timeslot) not in lecturer_times), None)
    if not lecturer_id:
        return None

    # Вибір груп з однаковим префіксом для лекцій
    if event_type == 'Лекція':
        # Витягуємо всі унікальні префікси
        prefixes = set(gid.split('-')[0] for gid in groups.keys())
        available_prefixes = []
        prefix_to_available_groups = {}
        for prefix in prefixes:
            available = [gid for gid in groups if gid.startswith(prefix) and (gid, timeslot) not in group_times]
            if available:
                available_prefixes.append(prefix)
                prefix_to_available_groups[prefix] = available
        if not available_prefixes:
            return None
        # Вибираємо випадковий префікс
        selected_prefix = random.choice(list(available_prefixes))
        available_groups = prefix_to_available_groups[selected_prefix]
        # Вибираємо кількість груп для лекції (від 1 до 3 або максимально доступні)
        num_groups = random.randint(1, min(3, len(available_groups)))
        group_ids = random.sample(available_groups, num_groups)
    else:
        group_id = subj['groupID']
        if (group_id, timeslot) in group_times:
            return None
        group_ids = [group_id]

    # Перевірка зайнятості груп
    for gid in group_ids:
        if (gid, timeslot) in group_times:
            return None

    # Перевірка зайнятості підгруп
    if event_type == 'Практика' and subj['requiresSubgroups']:
        if subgroup_ids is None:
            subgroup_ids = {gid: random.choice(groups[gid]['Subgroups']) for gid in group_ids}
        for gid, sgid in subgroup_ids.items():
            if (gid, sgid, timeslot) in subgroup_times:
                return None
    else:
        subgroup_ids = None

    # Вибір аудиторії з підходящою місткістю
    total_group_size = sum(
        groups[gid]['NumStudents'] // 2 if subgroup_ids and gid in subgroup_ids else groups[gid]['NumStudents']
        for gid in group_ids
    )
    suitable_auditoriums = [
        aid for aid, cap in auditoriums.items() if cap >= total_group_size
    ]
    if not suitable_auditoriums:
        return None

    # Вибір вільної аудиторії
    random.shuffle(suitable_auditoriums)
    auditorium_id = next((aid for aid in suitable_auditoriums if (aid, timeslot) not in auditorium_times), None)
    if not auditorium_id:
        return None

    event = Event(
        timeslot=timeslot,
        group_ids=group_ids,
        subject_id=subj['id'],
        subject_name=subj['name'],
        lecturer_id=lecturer_id,
        auditorium_id=auditorium_id,
        event_type=event_type,
        subgroup_ids=subgroup_ids,
        week_type=week_type
    )

    # Реєструємо зайнятість
    lecturer_times[(lecturer_id, timeslot)] = event
    for gid in group_ids:
        group_times[(gid, timeslot)] = event
        if event_type == 'Практика' and subgroup_ids and gid in subgroup_ids:
            sgid = subgroup_ids[gid]
            subgroup_times[(gid, sgid, timeslot)] = event
    auditorium_times[(auditorium_id, timeslot)] = event

    return event

def select_population(
    population: List[Schedule],
    groups: Dict[str, Dict],
    lecturers: Dict[str, Dict],
    auditoriums: Dict[str, int],
    fitness_function
) -> List[Schedule]:
    population.sort(key=lambda x: fitness_function(x, groups, lecturers, auditoriums))
    return population[:len(population) // 2] if len(population) > 1 else population

def herbivore_smoothing(
    population: List[Schedule],
    best_schedule: Schedule,
    lecturers: Dict[str, Dict],
    auditoriums: Dict[str, int],
    intensity: float = 0.1
) -> List[Schedule]:
    new_population: List[Schedule] = []
    for _ in range(len(population)):
        new_schedule = copy.deepcopy(best_schedule)
        mutate(new_schedule, lecturers, auditoriums, intensity)
        new_population.append(new_schedule)
    return new_population

def predator_approach(
    population: List[Schedule],
    groups: Dict[str, Dict],
    lecturers: Dict[str, Dict],
    auditoriums: Dict[str, int],
    fitness_function
) -> List[Schedule]:
    return select_population(population, groups, lecturers, auditoriums, fitness_function)

def rain(
    population_size: int,
    groups: Dict[str, Dict],
    subjects: List[Dict],
    lecturers: Dict[str, Dict],
    auditoriums: Dict[str, int]
) -> List[Schedule]:
    return generate_initial_population(population_size, groups, subjects, lecturers, auditoriums)

def mutate(
    schedule: Schedule,
    lecturers: Dict[str, Dict],
    auditoriums: Dict[str, int],
    intensity: float = 0.3
) -> None:
    num_events_to_mutate = max(2, int(len(schedule.events) * intensity))
    if num_events_to_mutate % 2 != 0:
        num_events_to_mutate += 1
    num_events_to_mutate = min(num_events_to_mutate, len(schedule.events) - (len(schedule.events) % 2))

    events_to_mutate = random.sample(schedule.events, num_events_to_mutate)
    for i in range(0, len(events_to_mutate), 2):
        event1, event2 = events_to_mutate[i], events_to_mutate[i + 1]

        if can_swap_events(event1, event2):
            # Обмін часовими слотами
            event1.timeslot, event2.timeslot = event2.timeslot, event1.timeslot

            # Обмін аудиторіями з імовірністю 50%
            if random.random() < 0.5 and can_swap_auditoriums(event1, event2):
                event1.auditorium_id, event2.auditorium_id = event2.auditorium_id, event1.auditorium_id

            # Обмін викладачами з імовірністю 50%
            if random.random() < 0.5 and can_swap_lecturers(event1, event2):
                event1.lecturer_id, event2.lecturer_id = event2.lecturer_id, event1.lecturer_id

def can_swap_events(event1: Event, event2: Event) -> bool:
    # Обмін можливий, якщо не виникає конфлікт типів занять для груп
    group_conflict = any(
        g in event2.group_ids for g in event1.group_ids
    ) and event1.event_type != event2.event_type
    return not group_conflict

def can_swap_auditoriums(event1: Event, event2: Event) -> bool:
    return event1.auditorium_id != event2.auditorium_id

def can_swap_lecturers(event1: Event, event2: Event) -> bool:
    return event1.lecturer_id != event2.lecturer_id

def soft_constraints_fitness(schedule: Schedule) -> int:
    return schedule.soft_constraints_score

def hard_constraints_fitness(schedule: Schedule) -> int:
    return schedule.hard_constraints_violations

def select_from_population(
    population: List[Schedule],
    fitness_function,
    n: int
) -> List[Schedule]:
    population.sort(key=fitness_function)
    return population[:n]

def crossover(parent1: Schedule, parent2: Schedule) -> Tuple[Schedule, Schedule]:
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    crossover_point = len(parent1.events) // 2

    child1.events[crossover_point:], child2.events[crossover_point:] = (
        parent2.events[crossover_point:], parent1.events[crossover_point:]
    )
    return child1, child2

def select_top_n(
    population: List[Schedule],
    fitness_function,
    n: int
) -> List[Schedule]:
    return select_from_population(population, fitness_function, n)

def genetic_algorithm(
    groups: Dict[str, Dict],
    subjects: List[Dict],
    lecturers: Dict[str, Dict],
    auditoriums: Dict[str, int],
    generations: int = 100
) -> Optional[Schedule]:
    population_size = 50
    n_best_to_select = 10
    population = generate_initial_population(population_size, groups, subjects, lecturers, auditoriums)

    # Етап 1: Жорсткі обмеження
    for generation in range(generations):
        population = select_top_n(population, hard_constraints_fitness, n_best_to_select)

        best_schedule = population[0]
        if best_schedule.hard_constraints_violations == 0:
            print(f"Покоління: {generation + 1}, Найкращий розклад для жорстких обмежень знайдено.")
            break
        else:
            print(f"Покоління: {generation + 1}, Найкращий розклад має {best_schedule.hard_constraints_violations} порушень жорстких обмежень.")

        # Схрещування
        new_population: List[Schedule] = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])

        # Мутація
        for schedule in new_population:
            if random.random() < 0.3:
                mutate(schedule, lecturers, auditoriums)

        population = new_population

    # Етап 2: Оптимізація м'яких обмежень
    for generation in range(generations):
        population = select_top_n(population, soft_constraints_fitness, n_best_to_select)

        best_schedule = population[0]
        best_fitness = best_schedule.soft_constraints_score
        print(f"Покоління: {generation + 1}, Оптимізація м'яких обмежень, поточна найкраща оцінка: {best_fitness}")

        if best_fitness == 0:
            print("Розклад оптимізовано без порушень м'яких обмежень.")
            break

        # Схрещування
        new_population: List[Schedule] = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])

        # Мутація
        for schedule in new_population:
            if random.random() < 0.3:
                mutate(schedule, lecturers, auditoriums)

        population = new_population

    return best_schedule

# Функція для виведення розкладу з додатковою інформацією
def print_schedule(schedule, lecturers, groups, auditoriums):
    schedule_dict = {}  # Створюємо словник для зберігання подій за часовими слотами
    for event in schedule.events:
        if event.timeslot not in schedule_dict:
            schedule_dict[event.timeslot] = []  # Ініціалізуємо список подій для нового часовго слота
        schedule_dict[event.timeslot].append(event)  # Додаємо подію до відповідного часовго слота

    # Словник для підрахунку годин викладачів
    lecturer_hours = {lecturer_id: 0 for lecturer_id in lecturers}

    # Виведення заголовків колонок
    print(f"{'Timeslot':<25} {'Group(s)':<30} {'Subject':<30} {'Type':<15} "
          f"{'Lecturer':<25} {'Auditorium':<10} {'Students':<10} {'Capacity':<10}")
    print("-" * 167)

    for timeslot in TIMESLOTS:
        if timeslot in schedule_dict:
            for event in schedule_dict[timeslot]:
                # Формуємо інформацію про групи, включаючи підгрупи, якщо вони є
                group_info = ', '.join([
                    f"{gid}" + (
                        f" (Subgroup {event.subgroup_ids[gid]})" if event.subgroup_ids and gid in event.subgroup_ids else ''
                    )
                    for gid in event.group_ids
                ])
                # Обчислюємо кількість студентів у події
                total_students = sum(
                    groups[g]['NumStudents'] // 2 if event.subgroup_ids and g in event.subgroup_ids else
                    groups[g]['NumStudents']
                    for g in event.group_ids
                )
                # Отримуємо місткість аудиторії
                auditorium_capacity = auditoriums[event.auditorium_id]

                # Виводимо інформацію по колонках
                print(f"{timeslot:<25} {group_info:<30} {event.subject_name:<30} {event.event_type:<15} "
                      f"{lecturers[event.lecturer_id]['lecturerName']:<25} {event.auditorium_id:<10} "
                      f"{total_students:<10} {auditorium_capacity:<10}")

                # Додаємо 1.5 години до загальної кількості годин викладача
                lecturer_hours[event.lecturer_id] += 1.5
        else:
            # Якщо у цьому часовому слоті немає подій, виводимо "EMPTY" у групах
            print(f"{timeslot:<25} {'EMPTY':<30} {'':<30} {'':<15} {'':<25} {'':<10} {'':<10} {'':<10}")
        print()  # Додаємо порожній рядок для відділення часових слотів

    # Виводимо кількість годин викладачів на тиждень
    print("\nКількість годин викладачів на тиждень:")
    print(f"{'Lecturer':<25} {'Total Hours':<10}")
    print("-" * 35)
    for lecturer_id, hours in lecturer_hours.items():
        lecturer_name = lecturers[lecturer_id]['lecturerName']
        print(f"{lecturer_name:<25} {hours:<10} годин")

def validate_data(groups, lecturers, subjects, auditoriums):
    # Перевірка, що всі groupID в subjects існують в groups
    for subj in subjects:
        if subj['groupID'] not in groups:
            raise ValueError(f"GroupID {subj['groupID']} in subject {subj['id']} does not exist in groups.")

    # Перевірка, що всі SubjectsCanTeach в lecturers існують в subjects
    subject_ids = {subj['id'] for subj in subjects}
    for lecturer_id, lecturer in lecturers.items():
        for subject_id in lecturer['SubjectsCanTeach']:
            if subject_id not in subject_ids:
                raise ValueError(f"SubjectID {subject_id} in lecturer {lecturer_id} does not exist in subjects.")

    # Перевірка, що всі аудиторії мають позитивну місткість
    for aid, cap in auditoriums.items():
        if cap <= 0:
            raise ValueError(f"Auditorium {aid} has non-positive capacity: {cap}")

def main():
    # Дані аудиторій
    auditoriums = {
        'A1': 30,
        'A2': 30,
        'A3': 30,
        'A4': 30,
        'B1': 30,
        'B2': 45,
        'B3': 60,
        'B4': 60,
    }

    # Дані груп
    groups = {
        'TTP-41': {
            'NumStudents': 30,
            'Subgroups': ['1', '2']
        },
        'TTP-42': {
            'NumStudents': 32,
            'Subgroups': ['1', '2']
        },
        'TK-41': {
            'NumStudents': 28,
            'Subgroups': ['1', '2']
        },
        'MI-41': {
            'NumStudents': 32,
            'Subgroups': ['1', '2']
        },
        'MI-42': {
            'NumStudents': 20,
            'Subgroups': ['1', '2']
        }
    }

    # Дані викладачів
    lecturers = {
        'L1': {
            'lecturerName': 'Мащенко С.О.',
            'SubjectsCanTeach': ['S1', 'S2'],
            'TypesCanTeach': ['Лекція', 'Практика'],
            'MaxHoursPerWeek': 20
        },
        'L2': {
            'lecturerName': 'Пашко А.О.',
            'SubjectsCanTeach': ['S2', 'S3'],
            'TypesCanTeach': ['Лекція', 'Практика'],
            'MaxHoursPerWeek': 18
        },
        'L3': {
            'lecturerName': 'Тарануха В.Ю.',
            'SubjectsCanTeach': ['S3', 'S4'],
            'TypesCanTeach': ['Лекція', 'Практика'],
            'MaxHoursPerWeek': 20
        },
        'L4': {
            'lecturerName': 'Ткаченко О.М.',
            'SubjectsCanTeach': ['S4', 'S5'],
            'TypesCanTeach': ['Лекція', 'Практика'],
            'MaxHoursPerWeek': 20
        },
        'L5': {
            'lecturerName': 'Шишацька О.В.',
            'SubjectsCanTeach': ['S5', 'S1'],
            'TypesCanTeach': ['Лекція', 'Практика'],
            'MaxHoursPerWeek': 20
        },
        'L6': {
            'lecturerName': 'Криволап А.В.',
            'SubjectsCanTeach': ['S4', 'S5'],
            'TypesCanTeach': ['Лекція', 'Практика'],
            'MaxHoursPerWeek': 20
        },
        'L7': {
            'lecturerName': 'Свистунов А.О.',
            'SubjectsCanTeach': ['S4', 'S5'],
            'TypesCanTeach': ['Лекція', 'Практика'],
            'MaxHoursPerWeek': 20
        },
        'L8': {
            'lecturerName': 'Зінько Т.П.',
            'SubjectsCanTeach': ['S1', 'S2'],
            'TypesCanTeach': ['Лекція', 'Практика'],
            'MaxHoursPerWeek': 20
        },
    }

    # Дані предметів
    subjects = [
        {
            'id': 'S1',
            'name': 'Теорія прийняття рішень',
            'groupID': 'TTP-41',
            'numLectures': 14,
            'numPracticals': 14,
            'requiresSubgroups': True,
            'weekType': 'Both'
        },
        {
            'id': 'S2',
            'name': 'Статистичне моделювання',
            'groupID': 'TTP-42',
            'numLectures': 14,
            'numPracticals': 14,
            'requiresSubgroups': True,
            'weekType': 'Both'
        },
        {
            'id': 'S3',
            'name': 'Інтелектуальні системи',
            'groupID': 'TK-41',
            'numLectures': 14,
            'numPracticals': 14,
            'requiresSubgroups': True,
            'weekType': 'Both'
        },
        {
            'id': 'S4',
            'name': 'Інформаційні технології',
            'groupID': 'MI-41',
            'numLectures': 14,
            'numPracticals': 14,
            'requiresSubgroups': True,
            'weekType': 'Both'
        },
        {
            'id': 'S5',
            'name': 'Розробка ПЗ під мобільні',
            'groupID': 'MI-42',
            'numLectures': 14,
            'numPracticals': 14,
            'requiresSubgroups': True,
            'weekType': 'Both'
        },
    ]

    # Валідація даних перед запуском алгоритму
    try:
        validate_data(groups, lecturers, subjects, auditoriums)
    except ValueError as e:
        print(f"Data validation error: {e}")
        return

    # Генерація розкладу
    best_schedule = genetic_algorithm(groups, subjects, lecturers, auditoriums, generations=100)

    # Вивід результату
    if best_schedule:
        print("\nНайкращий розклад знайдено:")
        print_schedule(best_schedule, lecturers, groups, auditoriums)
    else:
        print("Немає знайденого розкладу, що задовольняє всі обмеження.")

if __name__ == "__main__":
    main()