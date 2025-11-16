"""
Прототип системы умной визуализации данных биохимического анализатора
Цель: повышение эффективности и снижение ошибок в работе лаборатории
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings

# Настройка страницы
st.set_page_config(
    page_title="Умная система биохимического анализа",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация сессии
if 'analyses' not in st.session_state:
    st.session_state.analyses = []
if 'analyzer_connected' not in st.session_state:
    st.session_state.analyzer_connected = False
if 'quality_checks' not in st.session_state:
    st.session_state.quality_checks = []

# Нормальные значения показателей (расширенный список)
NORMAL_RANGES = {
    'Гемоглобин (Hb)': {'min': 120, 'max': 160, 'unit': 'г/л', 'gender_specific': True, 
                        'male': (130, 160), 'female': (120, 150), 'critical_low': 70, 'critical_high': 200},
    'Эритроциты (RBC)': {'min': 4.0, 'max': 5.5, 'unit': '×10¹²/л', 'gender_specific': True,
                         'male': (4.3, 5.7), 'female': (3.8, 5.1), 'critical_low': 2.0, 'critical_high': 7.0},
    'Лейкоциты (WBC)': {'min': 4.0, 'max': 9.0, 'unit': '×10⁹/л', 'critical_low': 1.0, 'critical_high': 30.0},
    'Тромбоциты (PLT)': {'min': 180, 'max': 320, 'unit': '×10⁹/л', 'critical_low': 50, 'critical_high': 1000},
    'Гематокрит (HCT)': {'min': 36, 'max': 48, 'unit': '%', 'gender_specific': True,
                        'male': (39, 49), 'female': (35, 45), 'critical_low': 20, 'critical_high': 60},
    'СОЭ': {'min': 2, 'max': 15, 'unit': 'мм/ч', 'gender_specific': True,
            'male': (2, 10), 'female': (2, 15), 'critical_low': 0, 'critical_high': 100},
    'Глюкоза': {'min': 3.9, 'max': 5.9, 'unit': 'ммоль/л', 'critical_low': 2.5, 'critical_high': 25.0},
    'Креатинин': {'min': 62, 'max': 106, 'unit': 'мкмоль/л', 'gender_specific': True,
                  'male': (80, 115), 'female': (53, 97), 'critical_low': 30, 'critical_high': 500},
    'Мочевина': {'min': 2.5, 'max': 8.3, 'unit': 'ммоль/л', 'critical_low': 1.0, 'critical_high': 50.0},
    'Билирубин общий': {'min': 3.4, 'max': 20.5, 'unit': 'мкмоль/л', 'critical_low': 0, 'critical_high': 200},
    'АЛТ': {'min': 10, 'max': 40, 'unit': 'Ед/л', 'gender_specific': True,
            'male': (10, 41), 'female': (7, 31), 'critical_low': 0, 'critical_high': 500},
    'АСТ': {'min': 10, 'max': 40, 'unit': 'Ед/л', 'gender_specific': True,
            'male': (10, 40), 'female': (10, 32), 'critical_low': 0, 'critical_high': 500},
    'Холестерин общий': {'min': 3.0, 'max': 5.2, 'unit': 'ммоль/л', 'critical_low': 1.0, 'critical_high': 10.0},
    'Белок общий': {'min': 65, 'max': 85, 'unit': 'г/л', 'critical_low': 40, 'critical_high': 120},
    'Альбумин': {'min': 35, 'max': 50, 'unit': 'г/л', 'critical_low': 20, 'critical_high': 70},
    'ЛДГ': {'min': 125, 'max': 220, 'unit': 'Ед/л', 'critical_low': 50, 'critical_high': 1000},
    'Щелочная фосфатаза': {'min': 40, 'max': 130, 'unit': 'Ед/л', 'gender_specific': True,
                           'male': (40, 130), 'female': (35, 105), 'critical_low': 10, 'critical_high': 500},
}

# Паттерны взаимосвязей между показателями
CORRELATION_PATTERNS = {
    'Анемия': {
        'indicators': ['Гемоглобин (Hb)', 'Эритроциты (RBC)', 'Гематокрит (HCT)'],
        'pattern': 'all_low',
        'severity': 'high'
    },
    'Воспаление': {
        'indicators': ['Лейкоциты (WBC)', 'СОЭ'],
        'pattern': 'both_high',
        'severity': 'medium'
    },
    'Печеночная недостаточность': {
        'indicators': ['АЛТ', 'АСТ', 'Билирубин общий'],
        'pattern': 'all_high',
        'severity': 'critical'
    },
    'Почечная недостаточность': {
        'indicators': ['Креатинин', 'Мочевина'],
        'pattern': 'both_high',
        'severity': 'critical'
    },
    'Сахарный диабет': {
        'indicators': ['Глюкоза'],
        'pattern': 'high',
        'severity': 'high'
    }
}

def get_normal_range(param_name, gender=None):
    """Получить нормальный диапазон для параметра"""
    if param_name not in NORMAL_RANGES:
        return None, None, None, None, ''
    
    param = NORMAL_RANGES[param_name]
    unit = param.get('unit', '')
    critical_low = param.get('critical_low')
    critical_high = param.get('critical_high')
    
    if param.get('gender_specific') and gender:
        if gender == 'Мужской':
            return param['male'][0], param['male'][1], critical_low, critical_high, unit
        else:
            return param['female'][0], param['female'][1], critical_low, critical_high, unit
    
    return param['min'], param['max'], critical_low, critical_high, unit

def check_abnormal(value, param_name, gender=None):
    """Проверить, является ли значение отклонением от нормы"""
    min_val, max_val, crit_low, crit_high, _ = get_normal_range(param_name, gender)
    if min_val is None:
        return None, None
    
    if value < crit_low or value > crit_high:
        return 'critical', 'critical'
    elif value < min_val:
        return 'low', 'abnormal'
    elif value > max_val:
        return 'high', 'abnormal'
    return 'normal', 'normal'

def detect_patterns(analysis_data, patient_gender=None):
    """Обнаружение паттернов и взаимосвязей в анализе"""
    detected_patterns = []
    
    for pattern_name, pattern_info in CORRELATION_PATTERNS.items():
        indicators = pattern_info['indicators']
        pattern_type = pattern_info['pattern']
        severity = pattern_info['severity']
        
        # Проверяем наличие всех индикаторов в данных
        if all(ind in analysis_data for ind in indicators):
            values = {ind: analysis_data[ind] for ind in indicators}
            statuses = {ind: check_abnormal(values[ind], ind, patient_gender)[0] 
                       for ind in indicators}
            
            match = False
            if pattern_type == 'all_low' and all(s == 'low' for s in statuses.values()):
                match = True
            elif pattern_type == 'all_high' and all(s == 'high' for s in statuses.values()):
                match = True
            elif pattern_type == 'both_high' and all(s in ['high', 'critical'] for s in statuses.values()):
                match = True
            elif pattern_type == 'high' and statuses[indicators[0]] in ['high', 'critical']:
                match = True
            
            if match:
                detected_patterns.append({
                    'name': pattern_name,
                    'severity': severity,
                    'indicators': indicators,
                    'values': values,
                    'statuses': statuses
                })
    
    return detected_patterns

def validate_data_quality(analysis_data, selected_params):
    """Валидация качества данных для снижения ошибок"""
    errors = []
    warnings_list = []
    
    # Проверка на отсутствующие значения
    missing = [p for p in selected_params if p not in analysis_data or analysis_data[p] is None]
    if missing:
        errors.append(f"Отсутствуют значения для параметров: {', '.join(missing)}")
    
    # Проверка на нереалистичные значения
    for param, value in analysis_data.items():
        if value is not None:
            _, _, crit_low, crit_high, _ = get_normal_range(param)
            if crit_low and value < crit_low * 0.1:  # Значение слишком низкое
                warnings_list.append(f"⚠️ {param}: значение {value} подозрительно низкое")
            if crit_high and value > crit_high * 10:  # Значение слишком высокое
                warnings_list.append(f"⚠️ {param}: значение {value} подозрительно высокое")
    
    # Проверка на логические несоответствия
    if 'Гемоглобин (Hb)' in analysis_data and 'Гематокрит (HCT)' in analysis_data:
        hb = analysis_data['Гемоглобин (Hb)']
        hct = analysis_data['Гематокрит (HCT)']
        if hb and hct:
            ratio = hb / (hct / 3) if hct > 0 else 0
            if ratio < 0.25 or ratio > 0.35:  # Нормальное соотношение Hb/Hct ≈ 0.3
                warnings_list.append("⚠️ Необычное соотношение Гемоглобин/Гематокрит")
    
    return errors, warnings_list

def calculate_trends(current_data, previous_data, selected_params):
    """Вычисление трендов по сравнению с предыдущим анализом"""
    trends = {}
    
    for param in selected_params:
        if param in current_data and param in previous_data:
            current = current_data[param]
            previous = previous_data[param]
            
            if current and previous:
                change = current - previous
                change_percent = (change / previous * 100) if previous != 0 else 0
                
                trends[param] = {
                    'current': current,
                    'previous': previous,
                    'change': change,
                    'change_percent': change_percent,
                    'direction': 'up' if change > 0 else 'down' if change < 0 else 'stable'
                }
    
    return trends

def create_smart_visualization(analysis_data, selected_params, patient_gender=None, 
                               previous_data=None, trends=None):
    """Создать умную визуализацию с инсайтами"""
    if not selected_params or not analysis_data:
        return None, []
    
    filtered_data = {k: v for k, v in analysis_data.items() if k in selected_params}
    
    if not filtered_data:
        return None, []
    
    params = []
    values = []
    colors = []
    units = []
    statuses = []
    min_vals = []
    max_vals = []
    insights = []
    
    for param_name, value in filtered_data.items():
        if value is not None:
            min_val, max_val, crit_low, crit_high, unit = get_normal_range(param_name, patient_gender)
            if min_val is not None:
                params.append(param_name)
                values.append(value)
                units.append(unit)
                min_vals.append(min_val)
                max_vals.append(max_val)
                
                status, severity = check_abnormal(value, param_name, patient_gender)
                statuses.append(status)
                
                # Цветовая кодировка с учетом критичности
                if severity == 'critical':
                    colors.append('#FF0000')  # Ярко-красный для критических
                    insights.append(f"🚨 КРИТИЧНО: {param_name} = {value:.2f} {unit}")
                elif status == 'high':
                    colors.append('#FF6666')  # Красный для повышенных
                elif status == 'low':
                    colors.append('#6666FF')  # Синий для пониженных
                else:
                    colors.append('#66FF66')  # Зеленый для нормальных
                
                # Добавляем информацию о тренде
                if trends and param_name in trends:
                    trend = trends[param_name]
                    if abs(trend['change_percent']) > 10:  # Значимое изменение
                        direction_icon = '📈' if trend['direction'] == 'up' else '📉'
                        insights.append(
                            f"{direction_icon} {param_name}: изменение на {trend['change_percent']:.1f}% "
                            f"({trend['previous']:.2f} → {trend['current']:.2f} {unit})"
                        )
    
    # Создаем график
    fig = go.Figure()
    
    # Добавляем зоны: критическая (красная), отклонение (желтая), норма (зеленая)
    for i, (param, min_v, max_v, crit_l, crit_h) in enumerate(zip(params, min_vals, max_vals, 
                                                                  [get_normal_range(p, patient_gender)[2] for p in params],
                                                                  [get_normal_range(p, patient_gender)[3] for p in params])):
        # Критическая зона
        if crit_l and crit_h:
            fig.add_trace(go.Scatter(
                x=[i, i, i, i],
                y=[crit_l, crit_h, crit_h, crit_l],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Нормальная зона
        fig.add_trace(go.Scatter(
            x=[i, i, i, i],
            y=[min_v, max_v, max_v, min_v],
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Добавляем значения
    fig.add_trace(go.Bar(
        x=params,
        y=values,
        marker_color=colors,
        marker_line=dict(color='white', width=2),
        text=[f"{v:.2f} {u}" for v, u in zip(values, units)],
        textposition='outside',
        name='Значение',
        hovertemplate='<b>%{x}</b><br>Значение: %{y:.2f}<br>Норма: %{customdata[0]:.2f} - %{customdata[1]:.2f} %{customdata[2]}<extra></extra>',
        customdata=[[m, M, u] for m, M, u in zip(min_vals, max_vals, units)]
    ))
    
    fig.update_layout(
        title={
            'text': '🧬 Умная визуализация биохимического анализа',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        xaxis_title='Параметры',
        yaxis_title='Значения',
        height=700,
        showlegend=False,
        xaxis=dict(tickangle=-45),
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig, insights

def simulate_analyzer_data(patient_gender='Мужской', include_abnormalities=True):
    """Симуляция данных от биохимического анализатора"""
    data = {}
    
    # Базовые значения в норме
    for param_name in NORMAL_RANGES.keys():
        min_val, max_val, _, _, unit = get_normal_range(param_name, patient_gender)
        if min_val and max_val:
            if include_abnormalities and np.random.random() < 0.3:  # 30% шанс отклонения
                if np.random.random() < 0.5:
                    # Пониженное значение
                    data[param_name] = min_val * (0.7 + np.random.random() * 0.2)
                else:
                    # Повышенное значение
                    data[param_name] = max_val * (1.2 + np.random.random() * 0.3)
            else:
                # Нормальное значение
                data[param_name] = min_val + (max_val - min_val) * np.random.random()
    
    return data

def main():
    st.title("🧬 Умная система биохимического анализа")
    st.markdown("**Прототип системы для повышения эффективности и снижения ошибок в работе лаборатории**")
    st.markdown("---")
    
    # Боковая панель
    mode = st.sidebar.radio(
        "Режим работы",
        ["🔌 Подключение анализатора", "👨‍🔬 Лаборант", "👨‍⚕️ Врач", "📊 Аналитика"]
    )
    
    if mode == "🔌 Подключение анализатора":
        analyzer_interface()
    elif mode == "👨‍🔬 Лаборант":
        smart_lab_interface()
    elif mode == "👨‍⚕️ Врач":
        smart_doctor_interface()
    else:
        analytics_interface()

def analyzer_interface():
    """Интерфейс подключения биохимического анализатора"""
    st.header("🔌 Подключение биохимического анализатора")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Статус подключения")
        if st.session_state.analyzer_connected:
            st.success("✅ Анализатор подключен")
            if st.button("🔌 Отключить"):
                st.session_state.analyzer_connected = False
                st.rerun()
        else:
            st.warning("⚠️ Анализатор не подключен")
            if st.button("🔌 Подключить (симуляция)"):
                st.session_state.analyzer_connected = True
                st.rerun()
        
        st.markdown("---")
        st.subheader("Автоматический ввод данных")
        
        if st.session_state.analyzer_connected:
            patient_gender = st.selectbox("Пол пациента", ["Мужской", "Женский"])
            include_abnormal = st.checkbox("Включить отклонения в тестовых данных", value=True)
            
            if st.button("📥 Получить данные от анализатора", type="primary"):
                data = simulate_analyzer_data(patient_gender, include_abnormal)
                st.session_state.analyzer_data = data
                st.session_state.analyzer_gender = patient_gender
                st.success("✅ Данные получены от анализатора!")
                st.json(data)
        else:
            st.info("Подключите анализатор для получения данных")
    
    with col2:
        st.subheader("Информация об анализаторе")
        st.info("""
        **Симуляция биохимического анализатора**
        
        В реальной системе здесь будет:
        - Подключение по протоколу HL7/ASCII
        - Автоматический импорт результатов
        - Валидация данных
        - Контроль качества
        """)
        
        if st.session_state.get('analyzer_data'):
            st.subheader("Последние полученные данные")
            st.json(st.session_state.analyzer_data)

def smart_lab_interface():
    """Улучшенный интерфейс лаборанта с умными функциями"""
    st.header("👨‍🔬 Интерфейс лаборанта (умная версия)")
    
    # Использование данных от анализатора, если доступны
    if st.session_state.get('analyzer_data') and st.session_state.analyzer_connected:
        st.info("💡 Используются данные от биохимического анализатора")
        if st.button("📋 Загрузить данные в форму"):
            st.session_state.use_analyzer_data = True
            st.rerun()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Информация о пациенте")
        patient_id = st.text_input("ID пациента", value="P-001")
        patient_name = st.text_input("ФИО пациента", value="Иванов Иван Иванович")
        gender_options = ["Мужской", "Женский"]
        default_gender = st.session_state.get('analyzer_gender', 'Мужской')
        default_index = gender_options.index(default_gender) if default_gender in gender_options else 0
        patient_gender = st.selectbox("Пол", gender_options, index=default_index)
        patient_age = st.number_input("Возраст", min_value=0, max_value=120, value=45)
        
        st.subheader("Тип анализа")
        is_stat = st.checkbox("СТАТ (срочный анализ)", value=False)
        priority = "СТАТ" if is_stat else "Обычный"
    
    with col2:
        st.subheader("Выбор параметров")
        all_params = list(NORMAL_RANGES.keys())
        
        # Группировка параметров
        basic = [p for p in all_params if p in ['Гемоглобин (Hb)', 'Эритроциты (RBC)', 
                                                'Лейкоциты (WBC)', 'Тромбоциты (PLT)', 
                                                'Гематокрит (HCT)', 'СОЭ']]
        biochemical = [p for p in all_params if p in ['Глюкоза', 'Креатинин', 'Мочевина', 
                                                      'Билирубин общий', 'Белок общий', 'Альбумин']]
        liver = [p for p in all_params if 'АЛТ' in p or 'АСТ' in p or 'ЛДГ' in p or 'Щелочная' in p]
        other = [p for p in all_params if p not in basic + biochemical + liver]
        
        selected_basic = st.multiselect("Основные показатели", basic, default=basic)
        selected_biochemical = st.multiselect("Биохимические", biochemical, default=biochemical)
        selected_liver = st.multiselect("Печеночные", liver, default=liver)
        selected_other = st.multiselect("Прочие", other, default=[])
        
        selected_params = selected_basic + selected_biochemical + selected_liver + selected_other
        st.session_state.selected_params = selected_params
    
    st.markdown("---")
    st.subheader("Ввод результатов анализа")
    
    # Автозаполнение из анализатора
    analysis_data = {}
    if st.session_state.get('use_analyzer_data') and st.session_state.get('analyzer_data'):
        analyzer_data = st.session_state.analyzer_data
        st.success("✅ Данные автозаполнены от анализатора")
    
    cols = st.columns(3)
    col_idx = 0
    
    for param in selected_params:
        min_val, max_val, _, _, unit = get_normal_range(param, patient_gender)
        normal_text = f"Норма: {min_val:.2f} - {max_val:.2f} {unit}" if min_val else ""
        
        # Автозаполнение значения
        default_value = None
        if st.session_state.get('use_analyzer_data') and st.session_state.get('analyzer_data'):
            default_value = analyzer_data.get(param, float((min_val + max_val) / 2) if min_val else 50.0)
        else:
            default_value = float((min_val + max_val) / 2) if min_val else 50.0
        
        with cols[col_idx % 3]:
            value = st.number_input(
                f"{param} ({unit})",
                min_value=0.0,
                value=default_value,
                step=0.1,
                help=normal_text,
                key=f"input_{param}"
            )
            analysis_data[param] = value
        col_idx += 1
    
    st.markdown("---")
    
    # Умная валидация
    errors, warnings = validate_data_quality(analysis_data, selected_params)
    
    if errors:
        st.error("❌ Ошибки валидации:")
        for error in errors:
            st.error(f"  • {error}")
    
    if warnings:
        st.warning("⚠️ Предупреждения:")
        for warning in warnings:
            st.warning(f"  • {warning}")
    
    # Обнаружение паттернов
    patterns = detect_patterns(analysis_data, patient_gender)
    if patterns:
        st.subheader("🔍 Обнаруженные паттерны")
        for pattern in patterns:
            severity_color = {'critical': '🔴', 'high': '🟠', 'medium': '🟡'}.get(pattern['severity'], '⚪')
            st.markdown(f"{severity_color} **{pattern['name']}** (серьезность: {pattern['severity']})")
            for ind, val in pattern['values'].items():
                st.write(f"  • {ind}: {val:.2f}")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💾 Сохранить анализ", type="primary", use_container_width=True):
            if not errors:
                save_analysis(patient_id, patient_name, patient_gender, patient_age, 
                            analysis_data, priority, selected_params)
                st.success("✅ Анализ сохранен!")
            else:
                st.error("Исправьте ошибки перед сохранением")
    
    with col2:
        if st.button("👁️ Предпросмотр визуализации", use_container_width=True):
            st.session_state.show_preview = True
    
    # Предпросмотр
    if st.session_state.get('show_preview', False):
        st.markdown("---")
        st.subheader("📊 Предпросмотр умной визуализации")
        
        # Получаем предыдущие данные для трендов
        previous_data = None
        patient_analyses = [a for a in st.session_state.analyses 
                          if a['patient_id'] == patient_id]
        if patient_analyses:
            previous_data = patient_analyses[-1]['data']
            trends = calculate_trends(analysis_data, previous_data, selected_params)
        else:
            trends = None
        
        fig, insights = create_smart_visualization(analysis_data, selected_params, 
                                                  patient_gender, previous_data, trends)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        if insights:
            st.subheader("💡 Умные инсайты")
            for insight in insights:
                st.markdown(f"  • {insight}")

def smart_doctor_interface():
    """Улучшенный интерфейс врача с умными функциями"""
    st.header("👨‍⚕️ Интерфейс врача (умная версия)")
    
    analyses = st.session_state.analyses
    if not analyses:
        st.warning("Нет доступных анализов")
        if st.button("📝 Загрузить демонстрационные данные"):
            create_demo_data()
            st.rerun()
        return
    
    sorted_analyses = sorted(analyses, key=lambda x: (x['priority'] != 'СТАТ', x['timestamp']), reverse=True)
    
    # Очередь с умными предупреждениями
    st.subheader("📋 Очередь анализов с умными предупреждениями")
    queue_data = []
    critical_patients = []
    
    for i, analysis in enumerate(sorted_analyses[:10]):
        # Проверка на критические значения
        critical_count = 0
        for param, value in analysis['data'].items():
            if value is not None:
                status, severity = check_abnormal(value, param, analysis['patient_gender'])
                if severity == 'critical':
                    critical_count += 1
        
        if critical_count > 0:
            critical_patients.append(analysis['patient_id'])
        
        priority_icon = "🔴" if analysis['priority'] == 'СТАТ' else "⚪"
        warning_icon = "🚨" if critical_count > 0 else ""
        
        queue_data.append({
            '№': i + 1,
            'Приоритет': f"{priority_icon} {analysis['priority']}",
            'Пациент': f"{warning_icon} {analysis['patient_name']}",
            'ID': analysis['patient_id'],
            'Критич.': critical_count,
            'Время': analysis['timestamp'].strftime('%H:%M:%S')
        })
    
    if queue_data:
        queue_df = pd.DataFrame(queue_data)
        st.dataframe(queue_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    patient_ids = sorted(list(set([a['patient_id'] for a in sorted_analyses])))
    selected_patient = st.selectbox("Выберите пациента", patient_ids)
    
    patient_analyses = [a for a in sorted_analyses if a['patient_id'] == selected_patient]
    if not patient_analyses:
        st.warning(f"Нет анализов для пациента {selected_patient}")
        return
    
    latest_analysis = patient_analyses[0]
    
    # Информация о пациенте
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Пациент", latest_analysis['patient_name'])
    with col2:
        st.metric("Пол", latest_analysis['patient_gender'])
    with col3:
        st.metric("Возраст", f"{latest_analysis['patient_age']} лет")
    with col4:
        priority_badge = "🔴 СТАТ" if latest_analysis['priority'] == 'СТАТ' else "⚪ Обычный"
        st.metric("Приоритет", priority_badge)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Текущий анализ", "📈 Тренды", "📜 История"])
    
    with tab1:
        display_smart_analysis(latest_analysis, patient_analyses)
    
    with tab2:
        display_trends_analysis(patient_analyses)
    
    with tab3:
        display_history(patient_analyses)

def display_smart_analysis(analysis, all_analyses):
    """Отображение анализа с умными функциями"""
    st.write(f"**Дата анализа:** {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Обнаружение паттернов
    patterns = detect_patterns(analysis['data'], analysis['patient_gender'])
    if patterns:
        st.subheader("🔍 Обнаруженные паттерны")
        for pattern in patterns:
            severity_icon = {'critical': '🚨', 'high': '⚠️', 'medium': 'ℹ️'}.get(pattern['severity'], '📌')
            st.markdown(f"{severity_icon} **{pattern['name']}**")
            with st.expander("Детали"):
                for ind, val in pattern['values'].items():
                    status_icon = {'high': '🔴', 'low': '🔵', 'normal': '✅'}.get(
                        pattern['statuses'][ind], '⚪')
                    st.write(f"{status_icon} {ind}: {val:.2f}")
    
    # Визуализация
    previous_data = all_analyses[1]['data'] if len(all_analyses) > 1 else None
    trends = calculate_trends(analysis['data'], previous_data, analysis['selected_params']) if previous_data else None
    
    fig, insights = create_smart_visualization(
        analysis['data'], 
        analysis['selected_params'], 
        analysis['patient_gender'],
        previous_data,
        trends
    )
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    if insights:
        st.subheader("💡 Умные инсайты и рекомендации")
        for insight in insights:
            st.markdown(f"  • {insight}")
    
    # Таблица статусов
    df = create_status_table(analysis['data'], analysis['selected_params'], analysis['patient_gender'])
    if df is not None:
        st.subheader("📋 Детальная таблица")
        st.dataframe(df, use_container_width=True, hide_index=True)

def display_trends_analysis(patient_analyses):
    """Отображение трендового анализа"""
    if len(patient_analyses) < 2:
        st.info("Недостаточно данных для анализа трендов (нужно минимум 2 анализа)")
        return
    
    st.subheader("📈 Анализ трендов")
    
    # Сортируем по дате
    sorted_by_date = sorted(patient_analyses, key=lambda x: x['timestamp'])
    
    # Выбираем параметры для трендового анализа
    common_params = set(sorted_by_date[0]['selected_params'])
    for analysis in sorted_by_date[1:]:
        common_params &= set(analysis['selected_params'])
    
    if not common_params:
        st.warning("Нет общих параметров для сравнения")
        return
    
    selected_trend_params = st.multiselect("Выберите параметры для трендового анализа", 
                                          list(common_params), default=list(common_params)[:5])
    
    if selected_trend_params:
        # Создаем график трендов
        fig = go.Figure()
        
        dates = [a['timestamp'] for a in sorted_by_date]
        
        for param in selected_trend_params:
            values = [a['data'].get(param) for a in sorted_by_date]
            min_val, max_val, _, _, unit = get_normal_range(param, sorted_by_date[0]['patient_gender'])
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name=param,
                hovertemplate=f'<b>{param}</b><br>Дата: %{{x}}<br>Значение: %{{y:.2f}} {unit}<extra></extra>'
            ))
            
            # Добавляем нормальный диапазон
            if min_val and max_val:
                fig.add_trace(go.Scatter(
                    x=[dates[0], dates[-1]],
                    y=[min_val, min_val],
                    mode='lines',
                    line=dict(color='gray', dash='dash', width=1),
                    name=f'{param} (мин)',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=[dates[0], dates[-1]],
                    y=[max_val, max_val],
                    mode='lines',
                    line=dict(color='gray', dash='dash', width=1),
                    name=f'{param} (макс)',
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title='Тренды показателей во времени',
            xaxis_title='Дата',
            yaxis_title='Значение',
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_history(patient_analyses):
    """Отображение истории анализов"""
    st.subheader("📜 История анализов")
    
    sorted_by_date = sorted(patient_analyses, key=lambda x: x['timestamp'], reverse=True)
    
    selected_date = st.selectbox(
        "Выберите дату анализа",
        [a['timestamp'].strftime('%Y-%m-%d %H:%M:%S') for a in sorted_by_date]
    )
    
    selected_analysis = next(a for a in sorted_by_date 
                            if a['timestamp'].strftime('%Y-%m-%d %H:%M:%S') == selected_date)
    
    display_smart_analysis(selected_analysis, patient_analyses)

def analytics_interface():
    """Интерфейс аналитики"""
    st.header("📊 Аналитика и статистика")
    
    analyses = st.session_state.analyses
    if not analyses:
        st.warning("Нет данных для аналитики")
        return
    
    # Общая статистика
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего анализов", len(analyses))
    with col2:
        stat_count = len([a for a in analyses if a['priority'] == 'СТАТ'])
        st.metric("СТАТ анализов", stat_count)
    with col3:
        total_abnormal = sum([len([v for v in a['data'].values() 
                                   if check_abnormal(v, list(a['data'].keys())[list(a['data'].values()).index(v)], 
                                                    a['patient_gender'])[1] != 'normal']) 
                              for a in analyses])
        st.metric("Отклонений обнаружено", total_abnormal)
    with col4:
        critical_count = sum([len([v for v in a['data'].values() 
                                  if check_abnormal(v, list(a['data'].keys())[list(a['data'].values()).index(v)], 
                                                   a['patient_gender'])[1] == 'critical']) 
                             for a in analyses])
        st.metric("Критических значений", critical_count, delta_color="inverse")
    
    st.markdown("---")
    
    # График распределения отклонений
    st.subheader("Распределение отклонений по параметрам")
    
    param_abnormalities = {}
    for analysis in analyses:
        for param, value in analysis['data'].items():
            if value is not None:
                status, severity = check_abnormal(value, param, analysis['patient_gender'])
                if severity != 'normal':
                    if param not in param_abnormalities:
                        param_abnormalities[param] = {'high': 0, 'low': 0, 'critical': 0}
                    if severity == 'critical':
                        param_abnormalities[param]['critical'] += 1
                    elif status == 'high':
                        param_abnormalities[param]['high'] += 1
                    elif status == 'low':
                        param_abnormalities[param]['low'] += 1
    
    if param_abnormalities:
        df_abn = pd.DataFrame(param_abnormalities).T
        df_abn = df_abn.sort_values('critical', ascending=False)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Критические', x=df_abn.index, y=df_abn['critical'], marker_color='red'))
        fig.add_trace(go.Bar(name='Повышенные', x=df_abn.index, y=df_abn['high'], marker_color='orange'))
        fig.add_trace(go.Bar(name='Пониженные', x=df_abn.index, y=df_abn['low'], marker_color='blue'))
        
        fig.update_layout(
            title='Распределение отклонений по параметрам',
            xaxis_title='Параметр',
            yaxis_title='Количество отклонений',
            barmode='stack',
            height=500,
            xaxis=dict(tickangle=-45)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_status_table(analysis_data, selected_params, patient_gender=None):
    """Создать таблицу со статусами параметров"""
    data = []
    for param in selected_params:
        if param in analysis_data and analysis_data[param] is not None:
            value = analysis_data[param]
            min_val, max_val, crit_low, crit_high, unit = get_normal_range(param, patient_gender)
            status, severity = check_abnormal(value, param, patient_gender)
            
            if severity == 'critical':
                status_text = '🚨 КРИТИЧНО'
                status_color = '#FF0000'
            elif status == 'high':
                status_text = '🔴 Повышен'
                status_color = '#FF6666'
            elif status == 'low':
                status_text = '🔵 Понижен'
                status_color = '#6666FF'
            else:
                status_text = '✅ Норма'
                status_color = '#66FF66'
            
            data.append({
                'Параметр': param,
                'Значение': f"{value:.2f} {unit}",
                'Норма': f"{min_val:.2f} - {max_val:.2f} {unit}" if min_val else 'N/A',
                'Статус': status_text
            })
    
    if not data:
        return None
    
    return pd.DataFrame(data)

def save_analysis(patient_id, patient_name, patient_gender, patient_age, analysis_data, priority, selected_params):
    """Сохранить анализ"""
    analysis = {
        'patient_id': patient_id,
        'patient_name': patient_name,
        'patient_gender': patient_gender,
        'patient_age': patient_age,
        'data': analysis_data,
        'priority': priority,
        'selected_params': selected_params,
        'timestamp': datetime.now()
    }
    
    st.session_state.analyses.append(analysis)
    save_to_file()

def save_to_file():
    """Сохранить анализы в файл"""
    try:
        analyses_json = []
        for a in st.session_state.analyses:
            a_copy = a.copy()
            a_copy['timestamp'] = a['timestamp'].isoformat()
            analyses_json.append(a_copy)
        
        with open('smart_analyses.json', 'w', encoding='utf-8') as f:
            json.dump(analyses_json, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Ошибка сохранения: {e}")

def load_from_file():
    """Загрузить анализы из файла"""
    try:
        if os.path.exists('smart_analyses.json'):
            with open('smart_analyses.json', 'r', encoding='utf-8') as f:
                analyses_json = json.load(f)
            
            analyses = []
            for a in analyses_json:
                a['timestamp'] = datetime.fromisoformat(a['timestamp'])
                analyses.append(a)
            
            st.session_state.analyses = analyses
    except Exception as e:
        pass

def create_demo_data():
    """Создать демонстрационные данные"""
    demo_analyses = [
        {
            'patient_id': 'P-001',
            'patient_name': 'Иванов Иван Иванович',
            'patient_gender': 'Мужской',
            'patient_age': 45,
            'priority': 'СТАТ',
            'selected_params': ['Гемоглобин (Hb)', 'Лейкоциты (WBC)', 'Тромбоциты (PLT)', 'Глюкоза', 'Креатинин'],
            'data': {
                'Гемоглобин (Hb)': 145.0,
                'Лейкоциты (WBC)': 12.5,
                'Тромбоциты (PLT)': 180.0,
                'Глюкоза': 6.2,
                'Креатинин': 95.0
            },
            'timestamp': datetime.now() - timedelta(minutes=5)
        },
        {
            'patient_id': 'P-001',
            'patient_name': 'Иванов Иван Иванович',
            'patient_gender': 'Мужской',
            'patient_age': 45,
            'priority': 'Обычный',
            'selected_params': ['Гемоглобин (Hb)', 'Лейкоциты (WBC)', 'Тромбоциты (PLT)', 'Глюкоза', 'Креатинин'],
            'data': {
                'Гемоглобин (Hb)': 140.0,
                'Лейкоциты (WBC)': 8.5,
                'Тромбоциты (PLT)': 200.0,
                'Глюкоза': 5.5,
                'Креатинин': 90.0
            },
            'timestamp': datetime.now() - timedelta(days=7)
        },
        {
            'patient_id': 'P-002',
            'patient_name': 'Петрова Мария Сергеевна',
            'patient_gender': 'Женский',
            'patient_age': 32,
            'priority': 'Обычный',
            'selected_params': ['Гемоглобин (Hb)', 'Эритроциты (RBC)', 'Лейкоциты (WBC)', 'СОЭ'],
            'data': {
                'Гемоглобин (Hb)': 115.0,
                'Эритроциты (RBC)': 3.5,
                'Лейкоциты (WBC)': 5.2,
                'СОЭ': 18.0
            },
            'timestamp': datetime.now() - timedelta(hours=2)
        },
        {
            'patient_id': 'P-003',
            'patient_name': 'Сидоров Петр Александрович',
            'patient_gender': 'Мужской',
            'patient_age': 58,
            'priority': 'СТАТ',
            'selected_params': ['Гемоглобин (Hb)', 'Лейкоциты (WBC)', 'АЛТ', 'АСТ', 'Билирубин общий'],
            'data': {
                'Гемоглобин (Hb)': 140.0,
                'Лейкоциты (WBC)': 8.5,
                'АЛТ': 65.0,
                'АСТ': 55.0,
                'Билирубин общий': 28.0
            },
            'timestamp': datetime.now() - timedelta(minutes=15)
        }
    ]
    
    st.session_state.analyses = demo_analyses
    save_to_file()

# Загрузка данных при старте
if len(st.session_state.analyses) == 0:
    load_from_file()

if __name__ == "__main__":
    main()

