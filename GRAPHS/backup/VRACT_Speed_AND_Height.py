import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# ==========================================
# ФИЗИЧЕСКИЕ КОНСТАНТЫ И ПАРАМЕТРЫ
# ==========================================
FILENAME = 'perfect_flight.csv'

# Планета (Кербин)
R_KERBIN  = 600000.0
MU_KERBIN = 3.5316e12

# Атмосфера Кербина (Блок 1: исправлено)
H_PRESSURE = 5600.0       # Масштабная высота давления, м [KSP Wiki]
P0         = 101325.0     # Давление у поверхности, Па
GAMMA      = 1.4
R_GAS      = 287.053      # Газовая постоянная, Дж/(кг·К)

# Двигатель LV-TX87 "Рысь" (Bobcat)
THRUST_VAC = 400000.0     # Тяга одного двигателя в вакууме, Н
THRUST_ATM = 374194.0     # Тяга одного двигателя у поверхности, Н
ISP_VAC    = 310.0        # Удельный импульс в вакууме, с
G0         = 9.80665      # Стандартное ускорение свободного падения, м/с²

# Расходы: из конфига двигателя  m_dot = F / (Isp * g0)
MDOT_S1    = THRUST_VAC * 5 / (ISP_VAC * G0)  # 5 двигателей, ~657.5 кг/с
MDOT_S2    = THRUST_VAC     / (ISP_VAC * G0)  # 1 двигатель,  ~131.5 кг/с

# Аэродинамика (оптимизировано по телеметрии)
CD_AREA_S1 = 4.708        # м², 1-я ступень (до разделения)
CD_AREA_S2 = 4.427        # м², 2-я ступень (после разделения)

# Тайминг миссии
SEP_TIME          = 61.5   # с — разделение ступеней
THROTTLE_DOWN_TIME = 75.5  # с — дросселирование (НЕ MECO!)
TRUE_MECO_TIME    = 109.5  # с — настоящее отключение двигателя (из телеметрии)

# Начальная масса 2-й ступени (из телеметрии, точка после сепарации)
MASS_S2_START = 22688.0   # кг

# Дроссель фазы 3: вычисляется аналитически, без подгонки
# MDOT_RESIDUAL = 14 кг/с измерено из телеметрии (dm = -7 кг / 0.5с = -14 кг/с)
MDOT_RESIDUAL  = 14.0                          # кг/с, измерено из телеметрии
THROTTLE_P3    = MDOT_RESIDUAL / MDOT_S2       # = 10.6% — тот же двигатель, меньший дроссель

# ==========================================
# АЭРОДИНАМИКА KSP (МНОЖИТЕЛЬ МАХА)
# ==========================================
MACH_POINTS = [0.0, 0.8, 1.05, 1.5, 2.0, 3.0, 5.0, 10.0]
MACH_MULTS  = [0.85, 0.85, 3.2, 1.6, 1.3, 1.1, 1.0, 1.0]

def get_mach_multiplier(mach):
    return np.interp(mach, MACH_POINTS, MACH_MULTS)

def get_atmosphere_properties(h):
    """
    Атмосфера Кербина (Блок 1):
      p     = P0 * exp(-h / H_PRESSURE)   [H=5600 м, KSP Wiki]
      rho   = p / (R_GAS * T)             [идеальный газ]
      a     = sqrt(GAMMA * R_GAS * T)
    Возвращает: (p_atm, rho, a)
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
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def extract_pitch_program(t_arr, h_arr, v_arr):
    h_smooth  = savgol_filter(h_arr, 11, 3)
    vy        = np.gradient(h_smooth, t_arr)
    v_safe    = np.where(v_arr > 1.0, v_arr, 1.0)
    sin_theta = np.clip(vy / v_safe, -1, 1)
    theta     = np.arcsin(sin_theta)
    return interp1d(t_arr, theta, kind='linear', fill_value="extrapolate")

def physics_ode(y, t, phase, pitch_func):
    """
    Система ОДУ. Четыре фазы:
      phase=1 — 5 двигателей, 100% (0 → SEP_TIME)
      phase=2 — 1 двигатель, 100% (SEP_TIME → THROTTLE_DOWN_TIME)
      phase=3 — 1 двигатель, ~10.6% коррекционный прожиг (THROTTLE_DOWN_TIME → TRUE_MECO_TIME)
      phase=4 — двигатели отключены, чистый коастинг (TRUE_MECO_TIME → конец)
    """
    h, v, m = y
    p_atm, rho, a = get_atmosphere_properties(h)
    g     = MU_KERBIN / (R_KERBIN + h)**2
    theta = float(pitch_func(t))

    thrust = 0.0
    dm     = 0.0

    if phase == 1:
        # 5 двигателей, полный газ
        thrust = THRUST_VAC*5 + (THRUST_ATM*5 - THRUST_VAC*5) * p_atm
        dm     = -MDOT_S1

    elif phase == 2:
        # 1 двигатель, полный газ
        thrust = THRUST_VAC + (THRUST_ATM - THRUST_VAC) * p_atm
        dm     = -MDOT_S2

    elif phase == 3:
        # Коррекционный прожиг: тот же двигатель на THROTTLE_P3 = 10.6%
        # Тяга пропорциональна дросселю, Isp тот же → новых параметров нет
        thrust = THRUST_VAC * THROTTLE_P3
        dm     = -MDOT_RESIDUAL

    elif phase == 4:
        # Настоящий MECO — двигатели отключены
        thrust = 0.0
        dm     = 0.0

    # Аэродинамика
    mach            = v / a if a > 0 else 0.0
    mach_mult       = get_mach_multiplier(mach)
    base_cd_area    = CD_AREA_S1 if phase == 1 else CD_AREA_S2
    current_cd_area = base_cd_area * mach_mult
    drag = 0.5 * rho * v**2 * current_cd_area

    dv_dt = (thrust - drag) / m - g * np.sin(theta)
    dh_dt = v * np.sin(theta)
    return [dh_dt, dv_dt, dm]

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

    pitch_program = extract_pitch_program(t_ksp, h_ksp, v_ksp)

    # Фаза 1 — 5 двигателей
    y0_1 = [h_ksp[0], max(0.1, v_ksp[0]), m_ksp[0]]
    t1   = np.linspace(0, SEP_TIME, 100)
    sol1 = odeint(physics_ode, y0_1, t1, args=(1, pitch_program))

    # Фаза 2 — 1 двигатель, полный газ
    y0_2 = [sol1[-1,0], sol1[-1,1], MASS_S2_START]
    t2   = np.linspace(SEP_TIME, THROTTLE_DOWN_TIME, 50)
    sol2 = odeint(physics_ode, y0_2, t2, args=(2, pitch_program))

    # Фаза 3 — коррекционный прожиг на 10.6%
    y0_3 = [sol2[-1,0], sol2[-1,1], sol2[-1,2]]
    t3   = np.linspace(THROTTLE_DOWN_TIME, TRUE_MECO_TIME, 100)
    sol3 = odeint(physics_ode, y0_3, t3, args=(3, pitch_program))

    # Фаза 4 — коастинг (настоящий MECO)
    y0_4 = [sol3[-1,0], sol3[-1,1], sol3[-1,2]]
    t4   = np.linspace(TRUE_MECO_TIME, 120, 50)
    sol4 = odeint(physics_ode, y0_4, t4, args=(4, pitch_program))

    t_model = np.concatenate([t1, t2[1:], t3[1:], t4[1:]])
    h_model = np.concatenate([sol1[:,0], sol2[1:,0], sol3[1:,0], sol4[1:,0]])
    v_model = np.concatenate([sol1[:,1], sol2[1:,1], sol3[1:,1], sol4[1:,1]])

    # --- График высоты ---
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(t_ksp,   h_ksp,   color='#ff7f0e', linewidth=4,   alpha=0.8, label='KSP (Телеметрия)')
    plt.plot(t_model, h_model, color='#1f77b4', linestyle='--', linewidth=2.5, label='Модель')
    plt.axvline(SEP_TIME,           color='gray',   linestyle=':', label='Разделение (61.5с)')
    plt.axvline(THROTTLE_DOWN_TIME, color='orange', linestyle=':', label='Дросселирование (75.5с)')
    plt.axvline(TRUE_MECO_TIME,     color='red',    linestyle=':', label='MECO (109.5с)')
    plt.title('Высота', fontsize=14, fontweight='bold')
    plt.ylabel('Высота, м'); plt.xlabel('Время, с')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig('Graph_Height.png')

    # --- График скорости ---
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(t_ksp,   v_ksp,   color='#2ca02c', linewidth=4,   alpha=0.8, label='KSP (Телеметрия)')
    plt.plot(t_model, v_model, color='#d62728', linestyle='--', linewidth=2.5, label='Модель')
    plt.axvline(SEP_TIME,           color='gray',   linestyle=':', label='Разделение (61.5с)')
    plt.axvline(THROTTLE_DOWN_TIME, color='orange', linestyle=':', label='Дросселирование (75.5с)')
    plt.axvline(TRUE_MECO_TIME,     color='red',    linestyle=':', label='MECO (109.5с)')
    plt.title('Скорость', fontsize=14, fontweight='bold')
    plt.ylabel('Скорость, м/с'); plt.xlabel('Время, с')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig('Graph_Speed.png')

    print("Графики сохранены: Graph_Height.png, Graph_Speed.png")
