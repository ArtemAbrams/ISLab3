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
            self._validate_hard_constraints(groups, lecturers, auditoriums)
            self._validate_soft_constraints(groups, lecturers, auditoriums)
            return self.hard_constraints_violations * 1000 + self.soft_constraints_score

        def _validate_hard_constraints(self, groups, lecturers, auditoriums):
            lecturer_times, group_times, auditorium_times = {}, {}, {}
            for event in self.events:
                # Лог перевірок
                print(f"Перевірка: {event.subject_name} для групи {event.group_ids} у {event.timeslot}")

                # Перевірка зайнятості викладача
                if (event.lecturer_id, event.timeslot) in lecturer_times:
                    print(f"Порушення: Викладач {event.lecturer_id} зайнятий у {event.timeslot}")
                    self.hard_constraints_violations += 1
                else:
                    lecturer_times[(event.lecturer_id, event.timeslot)] = event

                # Перевірка зайнятості груп
                for group_id in event.group_ids:
                    if (group_id, event.timeslot) in group_times:
                        print(f"Порушення: Група {group_id} зайнята у {event.timeslot}")
                        self.hard_constraints_violations += 1
                    else:
                        group_times[(group_id, event.timeslot)] = event

                # Перевірка зайнятості аудиторії
                if (event.auditorium_id, event.timeslot) in auditorium_times:
                    print(f"Порушення: Аудиторія {event.auditorium_id} зайнята у {event.timeslot}")
                    self.hard_constraints_violations += 1
                else:
                    auditorium_times[(event.auditorium_id, event.timeslot)] = event

        def _validate_soft_constraints(self, groups, lecturers, auditoriums):
            group_subject_hours = {gid: {} for gid in groups}
            for event in self.events:
                for group_id in event.group_ids:
                    if event.subject_id not in group_subject_hours[group_id]:
                        group_subject_hours[group_id][event.subject_id] = 0
                    if event.subgroup_ids and group_id in event.subgroup_ids:
                        group_subject_hours[group_id][event.subject_id] += 0.75  # По 0.75 години на підгрупу
                    else:
                        group_subject_hours[group_id][event.subject_id] += 1.5

            for group_id, subjects in group_subject_hours.items():
                for subject_id, hours in subjects.items():
                    required_hours = groups[group_id]['Subjects'][subject_id]['TotalHours']
                    if hours < required_hours:
                        self.soft_constraints_score += (required_hours - hours)

        def mutate(self, groups, lecturers, auditoriums):
            if random.random() < 0.5:
                self._remove_event()
            else:
                self._add_event(groups, lecturers, auditoriums)

        def _remove_event(self):
            if self.events:
                event_to_remove = random.choice(self.events)
                if event_to_remove.subgroup_ids:
                    self.events = [
                        e for e in self.events
                        if not (e.group_ids == event_to_remove.group_ids and e.timeslot == event_to_remove.timeslot)
                    ]
                else:
                    self.events.remove(event_to_remove)

        def _add_event(self, groups, lecturers, auditoriums):
            new_event = create_random_event(
                random.choice(list(groups.keys())),
                groups,
                lecturers,
                auditoriums
            )
            if new_event:
                self.add_event(new_event)

        def create_random_event(
                group_id: str,
                groups: Dict,
                lecturers: Dict,
                auditoriums: Dict
        ) -> Optional[Event]:
            suitable_subjects = groups[group_id]['Subjects']
            subject_id, subject = random.choice(list(suitable_subjects.items()))

            suitable_lecturers = [
                lid for lid, l in lecturers.items() if subject_id in l['SubjectsCanTeach']
            ]
            if not suitable_lecturers:
                return None

            lecturer_id = random.choice(suitable_lecturers)

            suitable_auditoriums = [
                aid for aid, cap in auditoriums.items() if cap >= groups[group_id]['NumStudents']
            ]
            if not suitable_auditoriums:
                return None

            auditorium_id = random.choice(suitable_auditoriums)

            timeslot = random.choice(TIMESLOTS)

            return Event(
                timeslot=timeslot,
                group_ids=[group_id],
                subject_id=subject_id,
                subject_name=subject['name'],
                lecturer_id=lecturer_id,
                auditorium_id=auditorium_id,
                event_type=random.choice(['Лекція', 'Практика'])
            )

def create_random_event(
        group_id: str,
        groups: Dict,
        lecturers: Dict,
        auditoriums: Dict
) -> Optional[Event]:
    suitable_subjects = groups[group_id]['Subjects']
    subject_id, subject = random.choice(list(suitable_subjects.items()))

    suitable_lecturers = [
        lid for lid, l in lecturers.items() if subject_id in l['SubjectsCanTeach']
    ]
    if not suitable_lecturers:
        return None

    lecturer_id = random.choice(suitable_lecturers)

    suitable_auditoriums = [
        aid for aid, cap in auditoriums.items() if cap >= groups[group_id]['NumStudents']
    ]
    if not suitable_auditoriums:
        return None

    auditorium_id = random.choice(suitable_auditoriums)

    timeslot = random.choice(TIMESLOTS)

    return Event(
        timeslot=timeslot,
        group_ids=[group_id],
        subject_id=subject_id,
        subject_name=subject['name'],
        lecturer_id=lecturer_id,
        auditorium_id=auditorium_id,
        event_type=random.choice(['Лекція', 'Практика'])
    )


def crossover(parent1: Schedule, parent2: Schedule) -> Tuple[Schedule, Schedule]:
    child1, child2 = Schedule(), Schedule()

    for timeslot in TIMESLOTS:
        if random.random() < 0.5:
            child1.events += [e for e in parent1.events if e.timeslot == timeslot]
            child2.events += [e for e in parent2.events if e.timeslot == timeslot]
        else:
            child1.events += [e for e in parent2.events if e.timeslot == timeslot]
            child2.events += [e for e in parent1.events if e.timeslot == timeslot]

    return child1, child2


def genetic_algorithm(groups, lecturers, auditoriums, generations=100, pop_size=50):
    population = generate_initial_population(pop_size, groups, lecturers, auditoriums)

    for generation in range(generations):
        population = sorted(population, key=lambda s: s.fitness(groups, lecturers, auditoriums))
        best = population[0]

        print(f"Покоління {generation + 1}/{generations}: Найкращий розклад - {len(best.events)} подій, "
              f"Порушень: {best.hard_constraints_violations}, Soft score: {best.soft_constraints_score}")

        if best.hard_constraints_violations == 0 and best.soft_constraints_score == 0:
            return best

        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(population[:10], 2)
            child1, child2 = crossover(parent1, parent2)
            child1.mutate(groups, lecturers, auditoriums)
            child2.mutate(groups, lecturers, auditoriums)
            new_population.extend([child1, child2])

        population = new_population[:pop_size]

    return population[0]


def generate_initial_population(
        pop_size: int,
        groups: Dict[str, Dict],
        lecturers: Dict[str, Dict],
        auditoriums: Dict[str, int]
) -> List[Schedule]:
    population = []
    for i in range(pop_size):
        schedule = Schedule()
        for group_id, group in groups.items():
            for subject_id, subject in group['Subjects'].items():
                num_events = int(subject['TotalHours'] // 1.5)  # 1.5 години на подію
                created_events = 0
                for _ in range(num_events):
                    new_event = create_random_event(group_id, groups, lecturers, auditoriums)
                    if new_event:
                        schedule.add_event(new_event)
                        created_events += 1
                print(
                    f"Генерація: Група {group_id}, Предмет {subject_id}, Заплановано занять: {num_events}, Створено: {created_events}")
        print(f"Розклад {i + 1}/{pop_size}: Подій створено {len(schedule.events)}")
        population.append(schedule)
    return population



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
        'A1': 100,
        'A2': 100
    }

    # Дані груп
    groups = {
        'G1': {
            'NumStudents': 20,
            'Subjects': {
                'S1': {'name': 'Math', 'TotalHours': 3},
            }
        },
        'G2': {
            'NumStudents': 30,
            'Subjects': {
                'S2': {'name': 'Physics', 'TotalHours': 3},
            }
        }
    }

    # Дані викладачів
    lecturers = {
        'L1': {
            'SubjectsCanTeach': ['S1', 'S2'],
            'MaxHoursPerWeek': 10
        }
    }

    print("Стартуємо генетичний алгоритм...")
    best_schedule = genetic_algorithm(groups, lecturers, auditoriums, generations=10, pop_size=5)

    if not best_schedule or not best_schedule.events:
        print("Не вдалося знайти розклад, що відповідає обмеженням.")
    else:
        print("Найкращий розклад знайдено:")
        print_schedule(best_schedule, lecturers, groups, auditoriums)

if __name__ == "__main__":
    main()