import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint

# ==========================================
# ФИЗИЧЕСКИЕ КОНСТАНТЫ И ПАРАМЕТРЫ
# ==========================================
FILENAME = 'perfect_flight.csv'

# Планета (Кербин)
R_KERBIN  = 600000.0
MU_KERBIN = 3.5316e12

# Атмосфера Кербина
H_PRESSURE = 5600.0       # Масштабная высота давления, м [KSP Wiki]
P0         = 101325.0     # Давление у поверхности, Па
GAMMA      = 1.4
R_GAS      = 287.053      # Газовая постоянная, Дж/(кг·К)

# Двигатель LV-TX87 "Рысь" (Bobcat)
THRUST_VAC = 400000.0     # Тяга одного двигателя в вакууме, Н
THRUST_ATM = 374194.0     # Тяга одного двигателя у поверхности, Н
ISP_VAC    = 310.0        # Удельный импульс в вакууме, с
G0         = 9.80665

MDOT_S1    = THRUST_VAC * 5 / (ISP_VAC * G0)  # 5 двигателей, ~657.5 кг/с
MDOT_S2    = THRUST_VAC     / (ISP_VAC * G0)  # 1 двигатель,  ~131.5 кг/с

# Аэродинамика (оптимизировано по телеметрии)
CD_AREA_S1 = 4.708        # м², 1-я ступень
CD_AREA_S2 = 4.427        # м², 2-я ступень

# Тайминг миссии
SEP_TIME           = 61.5   # с — разделение ступеней
THROTTLE_DOWN_TIME = 75.5   # с — дросселирование
TRUE_MECO_TIME     = 109.5  # с — настоящее отключение двигателя
T_END              = 120.0  # с — конец моделирования

# Начальная масса 2-й ступени (из телеметрии)
MASS_S2_START = 22688.0   # кг

# Коррекционный прожиг (фаза 3)
MDOT_RESIDUAL = 14.0                       # кг/с, из телеметрии
THROTTLE_P3   = MDOT_RESIDUAL / MDOT_S2   # = 10.6%

# ==========================================
# ПРОГРАММА ТАНГАЖА (gravity turn, без телеметрии)
# ==========================================
# Задаётся явно — θ не берётся из данных KSP.
# Параметры выбраны исходя из типичного профиля gravity turn для Кербина:
#   0–10с:       вертикальный старт (90°)
#   10–45с:      плавный поворот до ~60° (начало gravity turn)
#   45–SEP_TIME: доворот до ~45°
#   После сепарации: продолжение поворота до ~20° к MECO

PITCH_TIMES  = [0,    10,       45,          SEP_TIME,  THROTTLE_DOWN_TIME, TRUE_MECO_TIME, T_END]
PITCH_ANGLES = [90,   88,       65,          48,        35,                 20,             15]
# перевод в радианы
PITCH_RAD = np.radians(PITCH_ANGLES)

def pitch_program(t):
    """
    Явная программа тангажа — gravity turn без обращения к телеметрии.
    Угол от вертикали (90° = вверх, 0° = горизонт).
    Возвращает угол в радианах.
    """
    return float(np.interp(t, PITCH_TIMES, PITCH_RAD))

# ==========================================
# АЭРОДИНАМИКА KSP (МНОЖИТЕЛЬ МАХА)
# ==========================================
MACH_POINTS = [0.0, 0.8, 1.05, 1.5, 2.0, 3.0, 5.0, 10.0]
MACH_MULTS  = [0.85, 0.85, 3.2, 1.6, 1.3, 1.1, 1.0, 1.0]

def get_mach_multiplier(mach):
    return np.interp(mach, MACH_POINTS, MACH_MULTS)

def get_atmosphere(h):
    """
    Атмосфера Кербина:
      p   = P0 * exp(-h / H_PRESSURE)
      rho = p / (R_GAS * T)   [идеальный газ]
      a   = sqrt(GAMMA * R_GAS * T)
    """
    if h >= 70000:
        return 0.0, 0.0, 295.0
    p     = P0 * np.exp(-h / H_PRESSURE)
    p_atm = p / P0
    if h < 11000:
        T = 288.15 - 0.0065 * h
    elif h < 20000:
        T = 216.65
    else:
        T = 216.65 + 0.001 * (h - 20000)
    rho = p / (R_GAS * T)
    a   = np.sqrt(GAMMA * R_GAS * T)
    return p_atm, rho, a

# ==========================================
# 2D СИСТЕМА ОДУ
# ==========================================
# Вектор состояния: y = [x, y, vx, vy, m]
#   x, y   — декартовы координаты (начало = центр Кербина)
#   vx, vy — компоненты скорости
#   m      — масса
#
# Тяга направлена вдоль вектора скорости (pro-grade).
# Угол тангажа alpha — угол вектора скорости от горизонтали,
# вычисляется как alpha = atan2(vy, vx) с поправкой на pitch_program.



# --- пересобранная функция ---
def ode_2d(state, t, phase):
    """
    2D система ОДУ в декартовых координатах.
    state = [x, y_coord, vx, vy, m]
    Ось y — радиальная (от центра Кербина вверх).
    Ось x — тангенциальная (горизонталь).
    Тяга направлена по заданному углу тангажа pitch_program(t).
    """
    px, py, vx, vy, m = state

    r   = np.sqrt(px**2 + py**2)
    h   = r - R_KERBIN
    h   = max(h, 0.0)

    # Гравитация (направлена к центру планеты)
    g_mag = MU_KERBIN / r**2
    gx = -g_mag * px / r
    gy = -g_mag * py / r

    # Атмосфера
    p_atm, rho, a_snd = get_atmosphere(h)

    # Полная скорость
    v_mag = np.sqrt(vx**2 + vy**2)

    # Аэродинамика
    mach          = v_mag / a_snd if a_snd > 0 else 0.0
    mach_mult     = get_mach_multiplier(mach)
    base_cd_area  = CD_AREA_S1 if phase == 1 else CD_AREA_S2
    cd_area       = base_cd_area * mach_mult
    drag_mag      = 0.5 * rho * v_mag**2 * cd_area
    # Сопротивление направлено против скорости
    if v_mag > 0.01:
        drag_x = -drag_mag * vx / v_mag
        drag_y = -drag_mag * vy / v_mag
    else:
        drag_x = drag_y = 0.0

    # Тяга: угол задаётся pitch_program(t)
    # pitch_program(t) — угол от горизонтали в инерциальной СО
    # Направление "вверх" — единичный вектор от центра планеты
    # Направление "вправо" — перпендикуляр (тангенциальное)
    thrust_mag = 0.0
    dm         = 0.0

    if phase == 1:
        thrust_mag = THRUST_VAC*5 + (THRUST_ATM*5 - THRUST_VAC*5) * p_atm
        dm         = -MDOT_S1
    elif phase == 2:
        thrust_mag = THRUST_VAC + (THRUST_ATM - THRUST_VAC) * p_atm
        dm         = -MDOT_S2
    elif phase == 3:
        thrust_mag = THRUST_VAC * THROTTLE_P3
        dm         = -MDOT_RESIDUAL
    elif phase == 4:
        thrust_mag = 0.0
        dm         = 0.0

    # Единичные векторы в локальной СО
    # radial (вверх от поверхности)
    er_x = px / r;  er_y = py / r
    # tangential (перпендикуляр, направление "вперёд")
    et_x = -er_y;   et_y =  er_x

    alpha = pitch_program(t)   # угол от горизонтали
    thrust_x = thrust_mag * (np.cos(alpha) * et_x + np.sin(alpha) * er_x)
    thrust_y = thrust_mag * (np.cos(alpha) * et_y + np.sin(alpha) * er_y)

    # Уравнения движения
    ax = (thrust_x + drag_x) / m + gx
    ay = (thrust_y + drag_y) / m + gy

    return [vx, vy, ax, ay, dm]

# ==========================================
# ОСНОВНОЙ БЛОК
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(FILENAME):
        print(f"Файл {FILENAME} не найден.")
        exit()

    df    = pd.read_csv(FILENAME, sep=None, engine='python')
    t_ksp = df.iloc[:, 0].values * 0.5
    v_ksp = df.iloc[:, 1].values
    h_ksp = df.iloc[:, 2].values
    m_ksp = df.iloc[:, 3].values * 1000.0

    # Начальные условия: ракета стоит на поверхности Кербина
    # Начальная позиция — на оси y (тк для 2D)
    px0 = 0.0
    py0 = R_KERBIN + h_ksp[0]
    vx0 = max(0.1, v_ksp[0])   # небольшая горизонтальная скорость (gravity turn)
    vy0 = 0.0
    m0  = m_ksp[0]

    state0 = [px0, py0, vx0, vy0, m0]

    def run_phase(state_start, t_start, t_end, phase, n_steps=200):
        t_arr = np.linspace(t_start, t_end, n_steps)
        sol   = odeint(ode_2d, state_start, t_arr, args=(phase,))
        return t_arr, sol

    # Четыре фазы
    t1, s1 = run_phase(state0,    0,                  SEP_TIME,           1)
    s2_0   = list(s1[-1]); s2_0[4] = MASS_S2_START   # сброс 1-й ступени
    t2, s2 = run_phase(s2_0,      SEP_TIME,           THROTTLE_DOWN_TIME, 2)
    t3, s3 = run_phase(list(s2[-1]), THROTTLE_DOWN_TIME, TRUE_MECO_TIME,  3)
    t4, s4 = run_phase(list(s3[-1]), TRUE_MECO_TIME,     T_END,           4)

    # Объединяем
    t_all = np.concatenate([t1, t2[1:], t3[1:], t4[1:]])
    s_all = np.vstack([s1, s2[1:], s3[1:], s4[1:]])

    px_m = s_all[:, 0]
    py_m = s_all[:, 1]
    vx_m = s_all[:, 2]
    vy_m = s_all[:, 3]

    # Высота и полная скорость из 2D
    r_m     = np.sqrt(px_m**2 + py_m**2)
    h_model = r_m - R_KERBIN
    v_model = np.sqrt(vx_m**2 + vy_m**2)

    # ==========================================
    # ГРАФИКИ - СОХРАНЕНИЕ В ОТДЕЛЬНЫЕ ФАЙЛЫ
    # ==========================================
    
    # График высоты
    fig_height, ax_height = plt.subplots(figsize=(10, 6), dpi=150)
    ax_height.plot(t_ksp,  h_ksp,   color='#ff7f0e', linewidth=4, alpha=0.8, label='KSP (Телеметрия)')
    ax_height.plot(t_all,  h_model, color='#1f77b4', linestyle='--', linewidth=2.5, label='2D Модель')
    ax_height.axvline(SEP_TIME,           color='gray',   linestyle=':', label='Разделение (61.5с)')
    ax_height.axvline(THROTTLE_DOWN_TIME, color='orange', linestyle=':', label='Дросселирование (75.5с)')
    ax_height.axvline(TRUE_MECO_TIME,     color='red',    linestyle=':', label='MECO (109.5с)')
    ax_height.set_title('Высота — 2D', fontsize=13, fontweight='bold')
    ax_height.set_ylabel('Высота, м')
    ax_height.set_xlabel('Время, с')
    ax_height.grid(True, alpha=0.3)
    ax_height.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('Graph_Height_2D.png', dpi=150, bbox_inches='tight')
    plt.close(fig_height)
    print("График высоты сохранён: Graph_Height_2D.png")

    # График скорости
    fig_speed, ax_speed = plt.subplots(figsize=(10, 6), dpi=150)
    ax_speed.plot(t_ksp,  v_ksp,   color='#2ca02c', linewidth=4, alpha=0.8, label='KSP (Телеметрия)')
    ax_speed.plot(t_all,  v_model, color='#d62728', linestyle='--', linewidth=2.5, label='2D Модель')
    ax_speed.axvline(SEP_TIME,           color='gray',   linestyle=':', label='Разделение (61.5с)')
    ax_speed.axvline(THROTTLE_DOWN_TIME, color='orange', linestyle=':', label='Дросселирование (75.5с)')
    ax_speed.axvline(TRUE_MECO_TIME,     color='red',    linestyle=':', label='MECO (109.5с)')
    ax_speed.set_title('Скорость — 2D', fontsize=13, fontweight='bold')
    ax_speed.set_ylabel('Скорость, м/с')
    ax_speed.set_xlabel('Время, с')
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('Graph_Speed_2D.png', dpi=150, bbox_inches='tight')
    plt.close(fig_speed)
    print("График скорости сохранён: Graph_Speed_2D.png")

    print(f"\nКонечная высота: модель={h_model[-1]:.0f} м  телеметрия={h_ksp[-1]:.0f} м")
    print(f"Конечная скорость: модель={v_model[-1]:.1f} м/с  телеметрия={v_ksp[-1]:.1f} м/с")
