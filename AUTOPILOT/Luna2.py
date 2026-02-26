import krpc
import time
import math

# ------------------------------------------------------------------
# НАСТРОЙКИ ДЛЯ KSP STOCK (KERBIN)
# ------------------------------------------------------------------
TARGET_ORBIT_ALT = 100000.0  # Целевой апоцентр 100 км
ATMOSPHERE_HEIGHT = 70000.0  # Высота атмосферы Кербина


def get_target_pitch(altitude):
    """Профиль тангажа для гравитационного разворота."""
    if altitude < 1000:
        return 90
    if altitude > 60000:
        return 0
    frac = (altitude - 1000) / (60000 - 1000)
    return 90 * (1.0 - frac ** 0.6)


def main():
    conn = krpc.connect(name='Stock Luna-2')
    vessel = conn.space_center.active_vessel
    sc = conn.space_center

    # Потоки телеметрии
    ut = conn.add_stream(getattr, sc, 'ut')
    altitude = conn.add_stream(getattr, vessel.flight(), 'surface_altitude')
    apoapsis = conn.add_stream(getattr, vessel.orbit, 'apoapsis_altitude')
    periapsis = conn.add_stream(getattr, vessel.orbit, 'periapsis_altitude')

    print("[INFO] Скрипт запущен. Цель: апоцентр 100 км, затем угол 90° и остановка.")

    # 1. СТАРТ
    vessel.control.sas = False
    vessel.control.rcs = False
    vessel.control.throttle = 1.0

    ap = vessel.auto_pilot
    ap.reference_frame = vessel.surface_reference_frame
    ap.target_pitch_and_heading(90, 90)
    ap.engage()

    print(">>> 3.. 2.. 1.. ПУСК!")
    vessel.control.activate_next_stage()  # Зажигание двигателей
    time.sleep(0.5)
    if vessel.flight().surface_altitude < 5:
        vessel.control.activate_next_stage()  # Отделение клампов (если есть)

    # 2. ПОДЪЁМ (гравитационный разворот)
    phase = "ASCENT"
    print(">>> Фаза: НАБОР ВЫСОТЫ")
    target_ap = TARGET_ORBIT_ALT + 5000  # Целимся чуть выше (105 км) для запаса

    while phase == "ASCENT":
        alt = altitude()
        curr_ap = apoapsis()

        # Обновляем целевой тангаж
        pitch = get_target_pitch(alt)
        ap.target_pitch_and_heading(pitch, 90)

        # Управление тягой
        if curr_ap >= target_ap:
            if alt < ATMOSPHERE_HEIGHT:
                # В атмосфере – прикрываем, но не выключаем полностью
                vessel.control.throttle = 0.1
                if curr_ap > target_ap + 5000:
                    vessel.control.throttle = 0.0
            else:
                # Выше атмосферы – полное выключение
                vessel.control.throttle = 0.0
                phase = "COAST"
        else:
            vessel.control.throttle = 1.0

        # Автостейджинг (сброс пустых ступеней)
        if vessel.available_thrust < 1 and vessel.control.throttle > 0.5:
            print(f">>> Ступень сброшена! (Высота: {alt / 1000:.1f} км)")
            vessel.control.activate_next_stage()
            time.sleep(1.0)

        time.sleep(0.1)

    # 3. ПОЛЁТ ДО ВЫХОДА ИЗ АТМОСФЕРЫ
    print(">>> Фаза: ОЖИДАНИЕ ВЫХОДА ИЗ АТМОСФЕРЫ")
    while altitude() < ATMOSPHERE_HEIGHT:
        time.sleep(0.5)

    # 4. УСТАНОВКА УГЛА 90° (вертикально вверх)
    print(">>> Устанавливаю угол 90° (вертикально вверх)...")
    ap.reference_frame = vessel.surface_reference_frame
    ap.target_pitch_and_heading(90, 90)
    ap.engage()
    time.sleep(5)  # Даём время на ориентацию

    # Отключаем автопилот и включаем SAS для удержания
    ap.disengage()
    vessel.control.sas = True
    vessel.control.sas_mode = vessel.control.sas_mode.stability_assist
    print(">>> Скрипт завершён. Ракета на суборбитальной траектории с углом 90°, SAS активен.")
    # Скрипт заканчивается


if __name__ == '__main__':
    main()