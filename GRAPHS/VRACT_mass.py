import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# КОНФИГУРАЦИЯ И ФИЗИЧЕСКИЕ КОНСТАНТЫ
# ==========================================
FILENAME = 'perfect_flight.csv'
G0       = 9.80665         # Стандартное ускорение свободного падения, м/с²

# Параметры двигателя LV-TX87 "Рысь" (Bobcat)
THRUST_VAC  = 400000.0     # Тяга в вакууме, Н
ISP_VAC     = 310.0        # Удельный импульс в вакууме, с
MDOT_SINGLE = THRUST_VAC / (ISP_VAC * G0)  # ~131.5 кг/с (один двигатель)

# Конфигурация ступеней
NUM_ENGINES_S1 = 5         # 1-я ступень: 5 двигателей
NUM_ENGINES_S2 = 1         # 2-я ступень: 1 двигатель

# События полёта
SEP_TIME           = 61.5  # с — разделение ступеней
THROTTLE_DOWN_TIME = 75.5  # с — дросселирование (ошибочно называлось MECO)
TRUE_MECO_TIME     = 109.5 # с — настоящее отключение двигателя (из телеметрии)

# Начальная масса 2-й ступени (из телеметрии)
MASS_S2_WET = 22688.0      # кг

# Расход при коррекционном прожиге (фаза 3)
# Измерено из телеметрии: dm = -7 кг / 0.5с = -14 кг/с
MDOT_RESIDUAL = 14.0       # кг/с
THROTTLE_P3   = MDOT_RESIDUAL / MDOT_SINGLE  # = 10.6% дросселя

# ==========================================
# ФУНКЦИИ
# ==========================================
def load_flight_data(filepath):
    """Загружает данные из CSV."""
    if not os.path.exists(filepath):
        print(f"ОШИБКА: Файл {filepath} не найден.")
        return None, None
    df     = pd.read_csv(filepath, sep=None, engine='python')
    t_data = df.iloc[:, 0].values * 0.5
    m_data = df.iloc[:, 3].values * 1000.0
    return t_data, m_data

def simulate_mass_profile(t_array, start_mass):
    """
    Рассчитывает изменение массы ракеты. Четыре фазы:
      Фаза 1: 5 двигателей, 100%       (0 → SEP_TIME)
      Фаза 2: 1 двигатель, 100%        (SEP_TIME → THROTTLE_DOWN_TIME)
      Фаза 3: 1 двигатель, 10.6%       (THROTTLE_DOWN_TIME → TRUE_MECO_TIME)
      Фаза 4: двигатели выключены      (TRUE_MECO_TIME → конец)
    """
    mdot_s1 = MDOT_SINGLE * NUM_ENGINES_S1
    mdot_s2 = MDOT_SINGLE * NUM_ENGINES_S2

    # Масса в конце каждой фазы (для точки отсчёта следующей)
    mass_at_sep  = MASS_S2_WET
    mass_at_td   = mass_at_sep  - mdot_s2       * (THROTTLE_DOWN_TIME - SEP_TIME)
    mass_at_meco = mass_at_td   - MDOT_RESIDUAL * (TRUE_MECO_TIME - THROTTLE_DOWN_TIME)

    m_model = []
    for t in t_array:
        if t <= SEP_TIME:
            current_mass = start_mass - mdot_s1 * t
        elif t <= THROTTLE_DOWN_TIME:
            current_mass = mass_at_sep - mdot_s2 * (t - SEP_TIME)
        elif t <= TRUE_MECO_TIME:
            current_mass = mass_at_td - MDOT_RESIDUAL * (t - THROTTLE_DOWN_TIME)
        else:
            current_mass = mass_at_meco  # двигатели отключены
        m_model.append(current_mass)

    return np.array(m_model)

def plot_mass_comparison(t_ksp, m_ksp, t_model, m_model):
    """Строит график сравнения массы."""
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(t_ksp,   m_ksp,   color='#ff7f0e', linewidth=4,   alpha=0.8, label='KSP (Телеметрия)')
    plt.plot(t_model, m_model, color='#1f77b4', linestyle='--', linewidth=2.5, label='Расчётная модель')
    plt.axvline(SEP_TIME,           color='gray',   linestyle=':', label=f'Разделение ({SEP_TIME}с)')
    plt.axvline(THROTTLE_DOWN_TIME, color='orange', linestyle=':', label=f'Дросселирование ({THROTTLE_DOWN_TIME}с)')
    plt.axvline(TRUE_MECO_TIME,     color='red',    linestyle=':', label=f'MECO ({TRUE_MECO_TIME}с)')
    plt.title('Динамика изменения массы', fontsize=14, fontweight='bold')
    plt.ylabel('Масса, кг')
    plt.xlabel('Время, с')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Graph_Mass.png')
    print("График сохранён: Graph_Mass.png")

# ==========================================
# ОСНОВНОЙ БЛОК
# ==========================================
if __name__ == "__main__":
    t_ksp, m_ksp = load_flight_data(FILENAME)
    if t_ksp is None:
        exit()

    m_model = simulate_mass_profile(t_ksp, m_ksp[0])
    plot_mass_comparison(t_ksp, m_ksp, t_ksp, m_model)
