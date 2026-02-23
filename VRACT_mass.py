import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# КОНФИГУРАЦИЯ И ФИЗИЧЕСКИЕ КОНСТАНТЫ
# ==========================================
FILENAME = 'perfect_flight.csv'
G0 = 9.80665                # Ускорение свободного падения (стандарт), м/с²

# Параметры двигателя LV-TX87 "Рысь" (Bobcat)
THRUST_VAC = 400000.0       # Тяга в вакууме, Н
ISP_VAC = 310.0             # Удельный импульс в вакууме, с
# Расход одного двигателя: m_dot = F / (Isp * g0)
MDOT_SINGLE = THRUST_VAC / (ISP_VAC * G0)  # ~131.57 кг/с

# Конфигурация ступеней
NUM_ENGINES_S1 = 5          # 1 ступень: 5 двигателей
NUM_ENGINES_S2 = 1          # 2 ступень: 1 двигатель

# События полета (на основе анализа телеметрии)
SEP_TIME = 61.5             # Время разделения ступеней, с
MECO_TIME = 75.5            # Время отключения двигателя (Main Engine Cut-Off), с

# Масса после сброса 1 ступени (из данных KSP в момент t=62.0с)
MASS_S2_WET = 22687.0       # кг

def load_flight_data(filepath):
    """Загружает и подготавливает данные из CSV."""
    if not os.path.exists(filepath):
        print(f"ОШИБКА: Файл {filepath} не найден.")
        return None, None
    
    try:
        df = pd.read_csv(filepath, sep=None, engine='python')
        # В KSP часто запись идет тиками (0.5 сек), проверяем это
        t_data = df.iloc[:, 0].values * 0.5
        m_data = df.iloc[:, 3].values * 1000.0  # Перевод тонн в кг
        return t_data, m_data
    except Exception as e:
        print(f"ОШИБКА при чтении файла: {e}")
        return None, None

def simulate_mass_profile(t_array, start_mass):
    """
    Рассчитывает изменение массы ракеты во времени на основе
    расхода двигателей и событий разделения.
    """
    mdot_s1 = MDOT_SINGLE * NUM_ENGINES_S1
    mdot_s2 = MDOT_SINGLE * NUM_ENGINES_S2
    
    m_model = []
    
    # Расчет массы перед разделением для стыковки графиков
    # mass_at_sep_pre = start_mass - mdot_s1 * SEP_TIME

    for t in t_array:
        if t <= SEP_TIME:
            # Фаза 1: Работают 5 двигателей
            current_mass = start_mass - mdot_s1 * t
        elif t <= MECO_TIME:
            # Фаза 2: Работает 1 двигатель (после сброса ступени)
            # Мы используем зафиксированную массу MASS_S2_WET как точку отсчета
            dt_phase2 = t - SEP_TIME
            current_mass = MASS_S2_WET - mdot_s2 * dt_phase2
        else:
            # Фаза 3: Двигатели выключены, масса постоянна
            dt_burn_s2 = MECO_TIME - SEP_TIME
            current_mass = MASS_S2_WET - mdot_s2 * dt_burn_s2
            
        m_model.append(current_mass)
        
    return np.array(m_model)

def plot_mass_comparison(t_ksp, m_ksp, t_model, m_model):
    """Строит график сравнения массы."""
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Телеметрия
    plt.plot(t_ksp, m_ksp, color='#ff7f0e', linewidth=4, alpha=0.8, label='KSP (Телеметрия)')
    
    # Модель
    plt.plot(t_model, m_model, color='#1f77b4', linestyle='--', linewidth=2.5, label='Расчетная модель')
    
    # События
    plt.axvline(SEP_TIME, color='gray', linestyle=':', label=f'Разделение ({SEP_TIME}с)')
    plt.axvline(MECO_TIME, color='red', linestyle=':', label=f'MECO ({MECO_TIME}с)')
    
    plt.title('Динамика изменения массы (Расход топлива)', fontsize=14, fontweight='bold')
    plt.ylabel('Масса, кг', fontsize=12)
    plt.xlabel('Время, с', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Graph_1_Mass.png')
    print("График сохранен как 'Graph_1_Mass.png'")

# ==========================================
# ОСНОВНОЙ БЛОК ВЫПОЛНЕНИЯ
# ==========================================
if __name__ == "__main__":
    # 1. Загрузка данных
    t_ksp, m_ksp = load_flight_data(FILENAME)
    
    if t_ksp is not None:
        # 2. Моделирование
        # Создаем временную шкалу для модели (с запасом до 120с)
        t_model = np.linspace(0, 120, 500)
        m_model = simulate_mass_profile(t_model, start_mass=m_ksp[0])
        
        # 3. Отрисовка
        plot_mass_comparison(t_ksp, m_ksp, t_model, m_model)
