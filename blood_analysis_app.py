import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from streamlit_cookies_manager import EncryptedCookieManager

import json
import os

USERS = {
    "doctor1": {"password": "doc123", "role": "doctor"},
    "lab1": {"password": "lab123", "role": "lab"},
}
cookies = EncryptedCookieManager(
    prefix="blood_app_",  
    password="some-random-password-32chars!"  
)
# Настройка страницы
st.set_page_config(
    page_title="Система анализа крови",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = cookies.get("logged_in", "False") == "True"
    st.session_state.role = cookies.get("role", None)
    st.session_state.username = cookies.get("username", None)
    st.session_state.show_panel = cookies.get("show_panel", False)


if not cookies.ready():
    st.stop()

# Инициализация сессии
if 'analyses' not in st.session_state:
    st.session_state.analyses = []
if 'selected_params' not in st.session_state:
    st.session_state.selected_params = []

# Нормальные значения показателей крови
NORMAL_RANGES = {
    'Гемоглобин (Hb)': {'min': 120, 'max': 160, 'unit': 'г/л', 'gender_specific': True, 'male': (130, 160), 'female': (120, 150)},
    'Эритроциты (RBC)': {'min': 4.0, 'max': 5.5, 'unit': '×10¹²/л', 'gender_specific': True, 'male': (4.3, 5.7), 'female': (3.8, 5.1)},
    'Лейкоциты (WBC)': {'min': 4.0, 'max': 9.0, 'unit': '×10⁹/л'},
    'Тромбоциты (PLT)': {'min': 180, 'max': 320, 'unit': '×10⁹/л'},
    'Гематокрит (HCT)': {'min': 36, 'max': 48, 'unit': '%', 'gender_specific': True, 'male': (39, 49), 'female': (35, 45)},
    'СОЭ': {'min': 2, 'max': 15, 'unit': 'мм/ч', 'gender_specific': True, 'male': (2, 10), 'female': (2, 15)},
    'Глюкоза': {'min': 3.9, 'max': 5.9, 'unit': 'ммоль/л'},
    'Креатинин': {'min': 62, 'max': 106, 'unit': 'мкмоль/л', 'gender_specific': True, 'male': (80, 115), 'female': (53, 97)},
    'Мочевина': {'min': 2.5, 'max': 8.3, 'unit': 'ммоль/л'},
    'Билирубин общий': {'min': 3.4, 'max': 20.5, 'unit': 'мкмоль/л'},
    'АЛТ': {'min': 10, 'max': 40, 'unit': 'Ед/л', 'gender_specific': True, 'male': (10, 41), 'female': (7, 31)},
    'АСТ': {'min': 10, 'max': 40, 'unit': 'Ед/л', 'gender_specific': True, 'male': (10, 40), 'female': (10, 32)},
    'Холестерин общий': {'min': 3.0, 'max': 5.2, 'unit': 'ммоль/л'},
    'Белок общий': {'min': 65, 'max': 85, 'unit': 'г/л'},
    'Альбумин': {'min': 35, 'max': 50, 'unit': 'г/л'},
}

def get_normal_range(param_name, gender=None):
    """Получить нормальный диапазон для параметра"""
    if param_name not in NORMAL_RANGES:
        return None, None, ''
    
    param = NORMAL_RANGES[param_name]
    unit = param.get('unit', '')
    
    if param.get('gender_specific') and gender:
        if gender == 'Мужской':
            return param['male'][0], param['male'][1], unit
        else:
            return param['female'][0], param['female'][1], unit
    
    return param['min'], param['max'], unit

def check_abnormal(value, param_name, gender=None):
    """Проверить, является ли значение отклонением от нормы"""
    min_val, max_val, _ = get_normal_range(param_name, gender)
    if min_val is None:
        return None
    
    if value < min_val:
        return 'low'
    elif value > max_val:
        return 'high'
    return 'normal'

def create_visualization(analysis_data, selected_params, patient_gender=None):
    """Создать визуализацию результатов анализа"""
    if not selected_params or not analysis_data:
        return None
    
    # Фильтруем данные по выбранным параметрам
    filtered_data = {k: v for k, v in analysis_data.items() if k in selected_params}
    
    if not filtered_data:
        return None
    
    # Создаем данные для графика
    params = []
    values = []
    colors = []
    units = []
    statuses = []
    min_vals = []
    max_vals = []
    
    for param_name, value in filtered_data.items():
        if value is not None:
            min_val, max_val, unit = get_normal_range(param_name, patient_gender)
            if min_val is not None:
                params.append(param_name)
                values.append(value)
                units.append(unit)
                min_vals.append(min_val)
                max_vals.append(max_val)
                
                status = check_abnormal(value, param_name, patient_gender)
                statuses.append(status)
                
                if status == 'high':
                    colors.append('#FF4444')  # Красный для повышенных
                elif status == 'low':
                    colors.append('#4444FF')  # Синий для пониженных
                else:
                    colors.append('#44FF44')  # Зеленый для нормальных
    
    if not params:
        return None
    
    # Создаем интерактивный график
    fig = go.Figure()
    
    # Добавляем нормальные диапазоны (серые зоны)
    for i, (param, min_v, max_v) in enumerate(zip(params, min_vals, max_vals)):
        fig.add_trace(go.Scatter(
            x=[i, i, i, i],
            y=[min_v, max_v, max_v, min_v],
            fill='toself',
            fillcolor='rgba(200, 200, 200, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Добавляем значения параметров
    fig.add_trace(go.Bar(
        x=params,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f} {u}" for v, u in zip(values, units)],
        textposition='outside',
        name='Значение',
        hovertemplate='<b>%{x}</b><br>Значение: %{y:.2f}<br>Норма: %{customdata[0]:.2f} - %{customdata[1]:.2f} %{customdata[2]}<extra></extra>',
        customdata=[[m, M, u] for m, M, u in zip(min_vals, max_vals, units)]
    ))
    
    # Настройка графика
    fig.update_layout(
        title={
            'text': 'Результаты анализа крови',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        xaxis_title='Параметры',
        yaxis_title='Значения',
        height=600,
        showlegend=False,
        xaxis=dict(tickangle=-45),
        hovermode='closest'
    )
    
    return fig

def create_status_table(analysis_data, selected_params, patient_gender=None):
    """Создать таблицу со статусами параметров"""
    data = []
    for param in selected_params:
        if param in analysis_data and analysis_data[param] is not None:
            value = analysis_data[param]
            min_val, max_val, unit = get_normal_range(param, patient_gender)
            status = check_abnormal(value, param, patient_gender)
            
            if status == 'high':
                status_text = '🔴 Повышен'
                status_color = '#FF4444'
            elif status == 'low':
                status_text = '🔵 Понижен'
                status_color = '#4444FF'
            else:
                status_text = '✅ Норма'
                status_color = '#44FF44'
            
            data.append({
                'Параметр': param,
                'Значение': f"{value:.2f} {unit}",
                'Норма': f"{min_val:.2f} - {max_val:.2f} {unit}" if min_val else 'N/A',
                'Статус': status_text
            })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    return df

def login():
    st.title("🔐 Вход в систему")
    
    username = st.text_input("Имя пользователя")
    password = st.text_input("Пароль", type="password")
    
    if st.button("Войти"):
        user = USERS.get(username)
        if user and user["password"] == password:
            st.session_state.logged_in = True
            st.session_state.role = user["role"]
            st.session_state.username = username
            
            # Save login info in cookies
            cookies["logged_in"] = "True"
            cookies["role"] = user["role"]
            cookies["username"] = username
            cookies.save()
            
            st.success(f"Вы вошли как {username} ({user['role']})")
            
            # Instead of st.experimental_rerun, just set a flag to show panel
            st.session_state.show_panel = True
            st.session_state['refresh'] = not st.session_state.get('refresh', False)
            st.rerun()  
        else:
            st.error("Неверное имя пользователя или пароль")


    
def main():
    # If cookies/session say user is logged in, skip login
    if not st.session_state.get("logged_in", False):
        login()
        return
        # Show the correct panel based on role
    role = st.session_state.get("role")
    
    
    if role == "lab":
        lab_interface()
    else:
        doctor_interface()


def role_required(required_role):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if st.session_state.get("role") != required_role:
                st.warning("У вас нет доступа к этому разделу")
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator
    
def logout():
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None
    
    # Clear cookies
    cookies["logged_in"] = "False"
    cookies["role"] = ""
    cookies["username"] = ""
    cookies.save()
    st.rerun()  
    
@role_required("lab")
def lab_interface():
    """Интерфейс лаборанта"""
    col1, col2 = st.columns([9, 1])

    with col1:
        st.header("👨‍🔬 Интерфейс лаборанта")

    with col2:
        if st.button("🚪 Выйти"):
            logout()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Информация о пациенте")
        patient_id = st.text_input("ID пациента", value="P-001")
        patient_name = st.text_input("ФИО пациента", value="Иванов Иван Иванович")
        patient_gender = st.selectbox("Пол", ["Мужской", "Женский"])
        patient_age = st.number_input("Возраст", min_value=0, max_value=120, value=45)
        
        st.subheader("Тип анализа")
        is_stat = st.checkbox("СТАТ (срочный анализ)", value=False)
        priority = "СТАТ" if is_stat else "Обычный"
    
    with col2:
        st.subheader("Выбор параметров для анализа")
        st.info("Выберите параметры, которые необходимо отобразить врачу")
        
        # Группировка параметров
        basic_params = ['Гемоглобин (Hb)', 'Эритроциты (RBC)', 'Лейкоциты (WBC)', 'Тромбоциты (PLT)', 'Гематокрит (HCT)', 'СОЭ']
        biochemical_params = ['Глюкоза', 'Креатинин', 'Мочевина', 'Билирубин общий', 'Белок общий', 'Альбумин']
        liver_params = ['АЛТ', 'АСТ']
        lipid_params = ['Холестерин общий']
        
        st.write("**Основные показатели:**")
        selected_basic = st.multiselect("", basic_params, default=basic_params, label_visibility="collapsed")
        
        st.write("**Биохимические показатели:**")
        selected_biochemical = st.multiselect("", biochemical_params, default=biochemical_params, label_visibility="collapsed")
        
        st.write("**Печеночные ферменты:**")
        selected_liver = st.multiselect("", liver_params, default=liver_params, label_visibility="collapsed")
        
        st.write("**Липиды:**")
        selected_lipid = st.multiselect("", lipid_params, default=lipid_params, label_visibility="collapsed")
        
        selected_params = selected_basic + selected_biochemical + selected_liver + selected_lipid
        st.session_state.selected_params = selected_params
    
    st.markdown("---")
    st.subheader("Ввод результатов анализа")
    
    # Создаем форму для ввода значений
    analysis_data = {}
    cols = st.columns(3)
    col_idx = 0
    
    for param in selected_params:
        min_val, max_val, unit = get_normal_range(param, patient_gender)
        normal_text = f"Норма: {min_val:.2f} - {max_val:.2f} {unit}" if min_val else ""
        
        with cols[col_idx % 3]:
            value = st.number_input(
                f"{param} ({unit})",
                min_value=0.0,
                value=float((min_val + max_val) / 2) if min_val else 50.0,
                step=0.1,
                help=normal_text,
                key=f"input_{param}"
            )
            analysis_data[param] = value
        col_idx += 1
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💾 Сохранить анализ", type="primary", use_container_width=True):
            save_analysis(patient_id, patient_name, patient_gender, patient_age, analysis_data, priority, selected_params)
            st.success("✅ Анализ сохранен!")
    
    with col2:
        if st.button("👁️ Предпросмотр визуализации", use_container_width=True):
            st.session_state.show_preview = True
    
    # Предпросмотр визуализации
    if st.session_state.get('show_preview', False):
        st.markdown("---")
        st.subheader("📊 Предпросмотр визуализации")
        
        fig = create_visualization(analysis_data, selected_params, patient_gender)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        df = create_status_table(analysis_data, selected_params, patient_gender)
        if df is not None:
            st.subheader("📋 Таблица статусов")
            st.dataframe(df, use_container_width=True, hide_index=True)

@role_required("doctor")
def doctor_interface():
    """Интерфейс врача"""
    col1, col2 = st.columns([9, 1])

    with col1:
        st.header("👨‍⚕️ Интерфейс врача")

    with col2:
        if st.button("🚪 Выйти"):
            logout()
    
    # Получаем список анализов, отсортированных по приоритету (СТАТ первыми)
    analyses = st.session_state.analyses
    if not analyses:
        st.warning("Нет доступных анализов. Ожидание результатов...")
        if st.button("📝 Загрузить демонстрационные данные"):
            create_demo_data()
            st.rerun()
        return
    
    # Сортировка: сначала СТАТ, потом обычные, затем по времени
    sorted_analyses = sorted(analyses, key=lambda x: (x['priority'] != 'СТАТ', x['timestamp']), reverse=True)
    
    # Отображение очереди анализов
    st.subheader("📋 Очередь анализов")
    queue_data = []
    for i, analysis in enumerate(sorted_analyses[:10]):  # Показываем первые 10
        priority_icon = "🔴" if analysis['priority'] == 'СТАТ' else "⚪"
        queue_data.append({
            '№': i + 1,
            'Приоритет': f"{priority_icon} {analysis['priority']}",
            'Пациент': analysis['patient_name'],
            'ID': analysis['patient_id'],
            'Время': analysis['timestamp'].strftime('%H:%M:%S'),
            'Дата': analysis['timestamp'].strftime('%Y-%m-%d')
        })
    
    if queue_data:
        queue_df = pd.DataFrame(queue_data)
        st.dataframe(queue_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Выбор пациента
    patient_ids = sorted(list(set([a['patient_id'] for a in sorted_analyses])))
    selected_patient = st.selectbox("Выберите пациента", patient_ids)
    
    # Фильтрация анализов по пациенту
    patient_analyses = [a for a in sorted_analyses if a['patient_id'] == selected_patient]
    
    if not patient_analyses:
        st.warning(f"Нет анализов для пациента {selected_patient}")
        return
    
    # Информация о пациенте
    latest_analysis = patient_analyses[0]
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
    
    # Вкладки для текущего и исторических анализов
    tab1, tab2 = st.tabs(["📊 Текущий анализ", "📜 История анализов"])
    
    with tab1:
        display_analysis(latest_analysis)
    
    with tab2:
        st.subheader("История анализов")
        
        # Выбор анализа из истории
        analysis_dates = [a['timestamp'].strftime("%Y-%m-%d %H:%M:%S") for a in patient_analyses]
        selected_date = st.selectbox("Выберите дату анализа", analysis_dates)
        
        selected_analysis = next(a for a in patient_analyses if a['timestamp'].strftime("%Y-%m-%d %H:%M:%S") == selected_date)
        display_analysis(selected_analysis)

def display_analysis(analysis):
    """Отобразить анализ с визуализацией"""
    st.write(f"**Дата анализа:** {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**Приоритет:** {'🔴 СТАТ' if analysis['priority'] == 'СТАТ' else '⚪ Обычный'}")
    
    # Визуализация
    fig = create_visualization(
        analysis['data'], 
        analysis['selected_params'], 
        analysis['patient_gender']
    )
    if fig:
        # Fully unique key using patient ID + timestamp + id of the figure object
        chart_key = f"chart_{analysis['patient_id']}_{analysis['timestamp'].strftime('%Y%m%d%H%M%S')}_{id(fig)}"
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

    # Таблица статусов
    df = create_status_table(
        analysis['data'], 
        analysis['selected_params'], 
        analysis['patient_gender']
    )
    
    if df is not None:
        st.subheader("📋 Детальная таблица показателей")
        
        # Стилизация таблицы
        def highlight_status(val):
            if '🔴' in str(val):
                return 'background-color: #ffcccc'
            elif '🔵' in str(val):
                return 'background-color: #ccccff'
            elif '✅' in str(val):
                return 'background-color: #ffcccc'
            return ''
        
        styled_df = df.style.applymap(highlight_status, subset=['Статус'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Статистика
        col1, col2, col3 = st.columns(3)
        total = len(df)
        normal = len(df[df['Статус'].str.contains('✅')])
        high = len(df[df['Статус'].str.contains('🔴')])
        low = len(df[df['Статус'].str.contains('🔵')])
        
        with col1:
            st.metric("Всего параметров", total)
        with col2:
            st.metric("В норме", normal, delta=f"{normal/total*100:.1f}%")
        with col3:
            st.metric("Отклонения", high + low, delta=f"{(high+low)/total*100:.1f}%", delta_color="inverse")

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
    
    # Сохранение в файл (для персистентности)
    save_to_file()

def save_to_file():
    """Сохранить анализы в файл"""
    try:
        analyses_json = []
        for a in st.session_state.analyses:
            a_copy = a.copy()
            a_copy['timestamp'] = a['timestamp'].isoformat()
            analyses_json.append(a_copy)
        
        with open('blood_analyses.json', 'w', encoding='utf-8') as f:
            json.dump(analyses_json, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Ошибка сохранения: {e}")

def load_from_file():
    """Загрузить анализы из файла"""
    try:
        if os.path.exists('blood_analyses.json'):
            with open('blood_analyses.json', 'r', encoding='utf-8') as f:
                analyses_json = json.load(f)
            
            analyses = []
            for a in analyses_json:
                a['timestamp'] = datetime.fromisoformat(a['timestamp'])
                analyses.append(a)
            
            st.session_state.analyses = analyses
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")

def create_demo_data():
    """Создать демонстрационные данные для тестирования"""
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
                'Лейкоциты (WBC)': 12.5,  # Повышен
                'Тромбоциты (PLT)': 180.0,
                'Глюкоза': 6.2,  # Повышен
                'Креатинин': 95.0
            },
            'timestamp': datetime.now() - timedelta(minutes=5)
        },
        {
            'patient_id': 'P-002',
            'patient_name': 'Петрова Мария Сергеевна',
            'patient_gender': 'Женский',
            'patient_age': 32,
            'priority': 'Обычный',
            'selected_params': ['Гемоглобин (Hb)', 'Эритроциты (RBC)', 'Лейкоциты (WBC)', 'СОЭ'],
            'data': {
                'Гемоглобин (Hb)': 115.0,  # Понижен
                'Эритроциты (RBC)': 3.5,  # Понижен
                'Лейкоциты (WBC)': 5.2,
                'СОЭ': 18.0  # Повышен
            },
            'timestamp': datetime.now() - timedelta(hours=2)
        },
        {
            'patient_id': 'P-003',
            'patient_name': 'Сидоров Петр Александрович',
            'patient_gender': 'Мужской',
            'patient_age': 58,
            'priority': 'СТАТ',
            'selected_params': ['Гемоглобин (Hb)', 'Лейкоциты (WBC)', 'Тромбоциты (PLT)', 'АЛТ', 'АСТ', 'Билирубин общий'],
            'data': {
                'Гемоглобин (Hb)': 140.0,
                'Лейкоциты (WBC)': 8.5,
                'Тромбоциты (PLT)': 250.0,
                'АЛТ': 65.0,  # Повышен
                'АСТ': 55.0,  # Повышен
                'Билирубин общий': 28.0  # Повышен
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

