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
R_KERBIN = 600000.0         # Радиус планеты, м
MU_KERBIN = 3.5316e12       # Гравитационный параметр, м³/с²
ATM_SCALE_HEIGHT = 5000.0   # Высота однородной атмосферы, м
RHO_SURFACE = 1.225         # Плотность воздуха у поверхности, кг/м³

# Двигатели (Bobcat)
F_VAC_BASE = 400000.0       # Тяга вакуум (1 двиг), Н
F_ATM_BASE = 374194.0       # Тяга атмосфера (1 двиг), Н
MDOT_S1 = 657.65            # Расход 1 ступени (5 двиг), кг/с
MDOT_S2 = 131.53            # Расход 2 ступени (1 двиг), кг/с

# Аэродинамика
CD_AREA = 12.0              # Эффективная площадь сопротивления (Cd * A), м²

# Тайминг миссии
SEP_TIME = 61.5             # Разделение ступеней, с
MECO_TIME = 75.5            # Отключение двигателя, с
MASS_S2_START = 22687.0     # Масса 2 ступени в момент старта фазы 2, кг

# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def get_atmosphere_properties(h):
    """Возвращает давление (отн.) и плотность на заданной высоте."""
    if h < 70000:
        pressure_ratio = np.exp(-h / ATM_SCALE_HEIGHT)
        rho = RHO_SURFACE * pressure_ratio
    else:
        pressure_ratio = 0.0
        rho = 0.0
    return pressure_ratio, rho

def extract_pitch_program(t_arr, h_arr, v_arr):
    """
    Восстанавливает программу тангажа (Pitch) из телеметрии.
    Использует сглаживание для устранения шума производной.
    """
    # Сглаживание высоты фильтром Савицкого-Голея
    h_smooth = savgol_filter(h_arr, 11, 3)
    
    # Вертикальная скорость (dh/dt)
    vy = np.gradient(h_smooth, t_arr)
    
    # Защита от деления на ноль
    v_safe = np.where(v_arr > 1.0, v_arr, 1.0)
    
    # Sin(theta) = Vy / V_total
    sin_theta = np.clip(vy / v_safe, -1, 1)
    theta = np.arcsin(sin_theta)
    
    # Создаем интерполятор для использования в ODE
    return interp1d(t_arr, theta, kind='linear', fill_value="extrapolate")

def physics_ode(y, t, phase, pitch_func):
    """
    Система дифференциальных уравнений движения ракеты.
    y = [h, v, m] (Высота, Скорость, Масса)
    """
    h, v, m = y
    
    # 1. Параметры среды
    pressure_ratio, rho = get_atmosphere_properties(h)
    r = R_KERBIN + h
    g = MU_KERBIN / (r**2)
    
    # 2. Управление (угол тангажа)
    theta = pitch_func(t)
    
    # 3. Расчет тяги и расхода в зависимости от фазы
    thrust = 0.0
    dm = 0.0
    
    if phase == 1: # Работа 1 ступени (5 двигателей)
        thrust_vac_total = F_VAC_BASE * 5
        thrust_atm_total = F_ATM_BASE * 5
        # Тяга меняется от давления: F = F_vac + (F_atm - F_vac) * P_ratio
        # В KSP формула: F_actual = F_vac * efficiency. 
        # Здесь линейная интерполяция между F_vac и F_atm:
        current_thrust = thrust_vac_total + (thrust_atm_total - thrust_vac_total) * pressure_ratio
        thrust = current_thrust
        dm = -MDOT_S1
        
    elif phase == 2: # Работа 2 ступени (1 двигатель)
        if t <= MECO_TIME:
            current_thrust = F_VAC_BASE + (F_ATM_BASE - F_VAC_BASE) * pressure_ratio
            thrust = current_thrust
            dm = -MDOT_S2
        else:
            thrust = 0.0
            dm = 0.0
            
    # 4. Аэродинамическое сопротивление
    drag = 0.5 * rho * (v**2) * CD_AREA
    
    # 5. Уравнения движения (в проекции на вектор скорости)
    # dv/dt = (T - D)/m - g*sin(theta) + центробежная сила
    dv_dt = (thrust - drag) / m - (g - (v**2 / r)) * np.sin(theta)
    
    # dh/dt = v * sin(theta)
    dh_dt = v * np.sin(theta)
    
    return [dh_dt, dv_dt, dm]

# ==========================================
# ОСНОВНОЙ БЛОК
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(FILENAME):
        print(f"Файл {FILENAME} не найден.")
        exit()

    # 1. Загрузка данных
    df = pd.read_csv(FILENAME, sep=None, engine='python')
    t_ksp = df.iloc[:, 0].values * 0.5
    v_ksp = df.iloc[:, 1].values
    h_ksp = df.iloc[:, 2].values
    m_ksp = df.iloc[:, 3].values * 1000.0

    # 2. Подготовка модели управления
    pitch_program = extract_pitch_program(t_ksp, h_ksp, v_ksp)

    # 3. Интегрирование: Фаза 1 (0 -> SEP_TIME)
    y0_phase1 = [h_ksp[0], max(0.1, v_ksp[0]), m_ksp[0]]
    t1 = np.linspace(0, SEP_TIME, 100)
    
    sol1 = odeint(physics_ode, y0_phase1, t1, args=(1, pitch_program))

    # 4. Интегрирование: Фаза 2 (SEP_TIME -> 120)
    # Стыковка: берем высоту и скорость из конца 1 фазы, но массу сбрасываем
    last_state_p1 = sol1[-1]
    y0_phase2 = [last_state_p1[0], last_state_p1[1], MASS_S2_START]
    
    t2 = np.linspace(SEP_TIME, 120, 100)
    sol2 = odeint(physics_ode, y0_phase2, t2, args=(2, pitch_program))

    # 5. Объединение результатов
    t_model = np.concatenate([t1, t2[1:]])
    h_model = np.concatenate([sol1[:, 0], sol2[1:, 0]])

    # 6. Отрисовка графика высоты
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(t_ksp, h_ksp, color='#ff7f0e', linewidth=4, alpha=0.8, label='KSP (Телеметрия)')
    plt.plot(t_model, h_model, color='#1f77b4', linestyle='--', linewidth=2.5, label='Физ. модель')
    
    plt.axvline(SEP_TIME, color='gray', linestyle=':', label='Отстрел 1 ст.')
    plt.axvline(MECO_TIME, color='red', linestyle=':', label='MECO')

    plt.title('Сравнение высоты полета', fontsize=14, fontweight='bold')
    plt.ylabel('Высота, м', fontsize=12)
    plt.xlabel('Время, с', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Graph_2_Height.png')
    print("Graph_2_Height.png успешно сохранен.")
