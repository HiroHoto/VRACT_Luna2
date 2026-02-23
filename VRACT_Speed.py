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

# Планета и Атмосфера
R_KERBIN = 600000.0
MU_KERBIN = 3.5316e12
ATM_SCALE_HEIGHT = 5000.0
RHO_SURFACE = 1.225

# Силовая установка
F_VAC_BASE = 400000.0
F_ATM_BASE = 374194.0
MDOT_S1 = 657.65
MDOT_S2 = 131.53

# Аэродинамика
CD_AREA = 12.0  

# События
SEP_TIME = 61.5
MECO_TIME = 75.5
MASS_S2_START = 22687.0

# ==========================================
# ЛОГИКА МОДЕЛИРОВАНИЯ
# ==========================================

def get_atmosphere(h):
    if h < 70000:
        p_ratio = np.exp(-h / ATM_SCALE_HEIGHT)
        rho = RHO_SURFACE * p_ratio
    else:
        p_ratio = 0.0
        rho = 0.0
    return p_ratio, rho

def extract_pitch_program(t_arr, h_arr, v_arr):
    # Фильтрация шума высоты для корректного взятия производной
    h_smooth = savgol_filter(h_arr, 11, 3)
    vy = np.gradient(h_smooth, t_arr)
    v_safe = np.where(v_arr > 1.0, v_arr, 1.0)
    
    sin_theta = np.clip(vy / v_safe, -1, 1)
    theta = np.arcsin(sin_theta)
    
    return interp1d(t_arr, theta, kind='linear', fill_value="extrapolate")

def physics_ode(y, t, phase, pitch_func):
    """
    Основное уравнение динамики.
    Возвращает [dh/dt, dv/dt, dm/dt]
    """
    h, v, m = y
    p_ratio, rho = get_atmosphere(h)
    
    # Гравитация на текущей высоте
    r = R_KERBIN + h
    g = MU_KERBIN / (r**2)
    
    # Текущий угол тангажа
    theta = pitch_func(t)
    
    # Расчет тяги
    thrust = 0.0
    dm = 0.0
    
    if phase == 1:
        f_vac = F_VAC_BASE * 5
        f_atm = F_ATM_BASE * 5
        thrust = f_vac + (f_atm - f_vac) * p_ratio
        dm = -MDOT_S1
        
    elif phase == 2:
        if t <= MECO_TIME:
            thrust = F_VAC_BASE + (F_ATM_BASE - F_VAC_BASE) * p_ratio
            dm = -MDOT_S2
        else:
            thrust = 0.0
            dm = 0.0
            
    # Аэродинамика
    drag = 0.5 * rho * (v**2) * CD_AREA
    
    # Динамика скорости: Тяга - Сопротивление - (Гравитация - Центробежная)
    # (g - v^2/r) * sin(theta) — проекция "эффективной гравитации" на касательную траектории
    dv_dt = (thrust - drag) / m - (g - (v**2 / r)) * np.sin(theta)
    dh_dt = v * np.sin(theta)
    
    return [dh_dt, dv_dt, dm]

# ==========================================
# ВЫПОЛНЕНИЕ
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(FILENAME):
        print(f"Ошибка: {FILENAME} не найден.")
        exit()

    # 1. Данные
    df = pd.read_csv(FILENAME, sep=None, engine='python')
    t_ksp = df.iloc[:, 0].values * 0.5
    v_ksp = df.iloc[:, 1].values
    h_ksp = df.iloc[:, 2].values
    m_ksp = df.iloc[:, 3].values * 1000.0

    # 2. Модель управления
    pitch_program = extract_pitch_program(t_ksp, h_ksp, v_ksp)

    # 3. Интеграция
    # Phase 1
    y0 = [h_ksp[0], max(0.1, v_ksp[0]), m_ksp[0]]
    t1 = np.linspace(0, SEP_TIME, 150)
    sol1 = odeint(physics_ode, y0, t1, args=(1, pitch_program))

    # Phase 2 (сброс массы 1 ступени)
    y_trans = sol1[-1]
    y_start_2 = [y_trans[0], y_trans[1], MASS_S2_START]
    t2 = np.linspace(SEP_TIME, 120, 150)
    sol2 = odeint(physics_ode, y_start_2, t2, args=(2, pitch_program))

    # Объединение
    t_model = np.concatenate([t1, t2[1:]])
    v_model = np.concatenate([sol1[:, 1], sol2[1:, 1]])

    # 4. График
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(t_ksp, v_ksp, color='#ff7f0e', linewidth=4, alpha=0.8, label='KSP (Телеметрия)')
    plt.plot(t_model, v_model, color='#1f77b4', linestyle='--', linewidth=2.5, label='Физ. модель')
    
    plt.axvline(SEP_TIME, color='gray', linestyle=':', label='Отстрел 1 ст.')
    plt.axvline(MECO_TIME, color='red', linestyle=':', label='MECO')

    plt.title('Сравнение скорости (V_total)', fontsize=14, fontweight='bold')
    plt.ylabel('Скорость, м/с', fontsize=12)
    plt.xlabel('Время, с', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Graph_3_Speed.png')
    print("Graph_3_Speed.png успешно сохранен.")