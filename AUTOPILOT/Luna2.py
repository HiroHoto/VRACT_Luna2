import krpc
import time
import math


TARGET_ORBIT_ALT = 100000.0
ATMOSPHERE_HEIGHT = 70000.0


def get_target_pitch(altitude):
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

    ut = conn.add_stream(getattr, sc, 'ut')
    altitude = conn.add_stream(getattr, vessel.flight(), 'surface_altitude')
    apoapsis = conn.add_stream(getattr, vessel.orbit, 'apoapsis_altitude')
    periapsis = conn.add_stream(getattr, vessel.orbit, 'periapsis_altitude')

    print("[INFO] Скрипт запущен. Цель: апоцентр 100 км, затем угол 90° и остановка.")

    vessel.control.sas = False
    vessel.control.rcs = False
    vessel.control.throttle = 1.0

    ap = vessel.auto_pilot
    ap.reference_frame = vessel.surface_reference_frame
    ap.target_pitch_and_heading(90, 90)
    ap.engage()

    print(">>> 3.. 2.. 1.. ПУСК!")
    vessel.control.activate_next_stage()
    time.sleep(0.5)
    if vessel.flight().surface_altitude < 5:
        vessel.control.activate_next_stage()

    phase = "ASCENT"
    print(">>> Фаза: НАБОР ВЫСОТЫ")
    target_ap = TARGET_ORBIT_ALT + 5000

    while phase == "ASCENT":
        alt = altitude()
        curr_ap = apoapsis()

        pitch = get_target_pitch(alt)
        ap.target_pitch_and_heading(pitch, 90)

        if curr_ap >= target_ap:
            if alt < ATMOSPHERE_HEIGHT:
                vessel.control.throttle = 0.1
                if curr_ap > target_ap + 5000:
                    vessel.control.throttle = 0.0
            else:
                vessel.control.throttle = 0.0
                phase = "COAST"
        else:
            vessel.control.throttle = 1.0

        if vessel.available_thrust < 1 and vessel.control.throttle > 0.5:
            print(f">>> Ступень сброшена! (Высота: {alt / 1000:.1f} км)")
            vessel.control.activate_next_stage()
            time.sleep(1.0)

        time.sleep(0.1)

    print(">>> Фаза: ОЖИДАНИЕ ВЫХОДА ИЗ АТМОСФЕРЫ")
    while altitude() < ATMOSPHERE_HEIGHT:
        time.sleep(0.5)

    print(">>> Устанавливаю угол 90° (вертикально вверх)...")
    ap.reference_frame = vessel.surface_reference_frame
    ap.target_pitch_and_heading(90, 90)
    ap.engage()
    time.sleep(5)

    ap.disengage()
    vessel.control.sas = True
    vessel.control.sas_mode = vessel.control.sas_mode.stability_assist
    print(">>> Скрипт завершён. Ракета на суборбитальной траектории с углом 90°, SAS активен.")


if __name__ == '__main__':
    main()