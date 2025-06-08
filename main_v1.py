#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stroke Rehabilitation Assistant Application - Dialog Language & UI Fix v2
-----------------------------------------------------------------------
Changes:
- Fixed exercise names in SelectionDialog not translating.
- Ensured language button in SelectionDialog remains consistent after
  language changes in PatientDialog by refactoring UI updates.
- Fixed button text superposition in MainWindow after language change.
- Added language switching capability within PatientDialog and SelectionDialog.
- Dialogs now reflect and can update a shared application language state.
- PatientDialog for selecting or creating a patient profile.
- Session reports are now saved in patient-specific subfolders: output/patient_name/
- Added a placeholder "Create Summary" button on the finished screen.
- Implemented language localization for Markdown report headers and content.

Requirements:
- PySide6, pyqtgraph, pyserial, matplotlib (to be used for report_generator.py)
- Sound files (WAV) in 'sounds/' subdirectory.
- Video files (MP4) in 'videos/' subdirectory.
- Image files (PNG) in 'images/' subdirectory.
- Reports (MD) in 'output/patient_name/' subdirectory.
"""

import sys
import time
import random
# import queue
# import threading
import datetime
import os
from platformdirs import user_data_dir
import shutil  # For copying files
from emg_worker_threaded import EMGProcessingWorker, EMGDataAcquisitionWorker

from report_generator import generate_summary_report_for_patient

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QStackedWidget, QDialog, QMessageBox,
    QLineEdit, QComboBox
)
from PySide6.QtMultimedia import QMediaPlayer, QSoundEffect
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import QTimer, QUrl, Qt, Slot, Signal, QObject, QThread
from PySide6.QtGui import QColor, QPalette, QPixmap, QIcon, QDesktopServices

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QPushButton, QLabel, QComboBox, QCheckBox, QHBoxLayout, QFrame, QFileDialog

import pyqtgraph as pg
import serial

import json
import threading

def run_application(existing_app=None):
    """
    Run the application with proper QApplication management
    
    Args:
        existing_app: An existing QApplication instance if available
        
    Returns:
        The exit code from the application
    """
    import sys
    from PyQt5.QtWidgets import QApplication
    import pyqtgraph as pg
    from PyQt5.QtGui import QIcon
    import os
    
    # Use existing app or create a new one
    if existing_app:
        app = existing_app
    else:
        app = QApplication.instance() or QApplication(sys.argv)
    
    # Configure application
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    app.setWindowIcon(QIcon(IMAGE_PATH + "app_icon.png"))
    
    # Ensure the base OUTPUT_PATH exists at startup
    try:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        print(f"Ensured base output directory exists: {OUTPUT_PATH}")
    except OSError as e:
        print(f"Could not create base output directory {OUTPUT_PATH}: {e}")
    
    # Set application properties
    app.current_app_language = DEFAULT_LANGUAGE
    app.main_window_instance = None
    current_patient_name = None
    main_window_ref = None
    
    # Main application loop
    while True:
        current_lang_for_dialogs = app.current_app_language
        
        if not current_patient_name:
            patient_dialog = PatientDialog()
            if patient_dialog.exec() != QDialog.Accepted or not patient_dialog.selected_patient_name:
                print("Patient selection cancelled or no name provided. Exiting.")
                return 0
            current_patient_name = patient_dialog.selected_patient_name
            app.current_app_language = patient_dialog.dialog_language
            
        current_lang_for_dialogs = app.current_app_language
        exercise_dialog = SelectionDialog(patient_name=current_patient_name)
        
        if exercise_dialog.exec() != QDialog.Accepted:
            print("Exercise selection dialog closed or rejected. Exiting.")
            return 0
            
        app.current_app_language = exercise_dialog.dialog_language
        
        if main_window_ref is not None:
            print("Deleting previous main window instance.")
            main_window_ref.deleteLater()
            main_window_ref = None
            app.main_window_instance = None
            
        main_window_ref = MainWindow(
            patient_name=current_patient_name,
            selected_sequence_name=exercise_dialog.selected_sequence_name_internal,
            selected_steps=exercise_dialog.selected_sequence_steps
        )
        
        app.main_window_instance = main_window_ref
        main_window_ref.show()
        
        # This is the original issue - we should only have one exec() call
        # and handle restarting differently
        app.processEvents()  # Process pending events
        result = app.exec_()
        
        # Check if we need to restart or exit
        if hasattr(main_window_ref, 'close_for_restart_flag') and main_window_ref.close_for_restart_flag:
            print("Restarting application flow (Play Again was clicked).")
            # No need to exit the event loop, just continue the while loop
            continue
        else:
            print("Application will now exit.")
            return result


# --- Configuration ---
UPDATE_INTERVAL_MS = 100
ADVANCE_DELAY_MS = 1000 # Time that it takes to move to next movement after CORRECT_MOVEMENT
SHORT_DELAY_FOR_STRONG_FEEDBACK_MS = 1000 
SERIAL_PORT = "COM3" # Serial Port that Connects to Arduino
BAUD_RATE = 9600 # Correspond to the Baud Rate (taxa transmissão) of Arduino Code
# --- Don't Change Further Code

APP_TITLE = "Stroke Rehabilitation Assistant"
LANGUAGES = ['en', 'pt']
DEFAULT_LANGUAGE = 'en'

# EMG Simulation
SIMULATE_EMG = True
SIMULATION_DELAY_S = 2 # Time between New Simulated Signal
POSSIBLE_STATUSES = ['NO_MOVEMENT', 'INCORRECT_MOVEMENT', 'CORRECT_WEAK', 'CORRECT_STRONG']
STATUS_WEIGHTS = [0.3, 0.2, 0.3, 0.2]

if not QApplication.instance():
    app = QApplication(sys.argv)
else:
    app = QApplication.instance()


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        # For one-directory builds, _MEIPASS is the path to the bundle's root (_internal folder)
        base_path = sys._MEIPASS
    except AttributeError:
        # _MEIPASS is not set
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Resource Paths ---
VIDEO_PATH = resource_path("videos/")
SOUND_PATH = resource_path("sounds/")
IMAGE_PATH = resource_path("images/")

APP_NAME = "StrokeRehabApp"
APP_AUTHOR = "MioBuddies"
user_specific_data_path = user_data_dir(APP_NAME, APP_AUTHOR, roaming=False)
OUTPUT_PATH = os.path.join(user_specific_data_path, "output")
os.makedirs(OUTPUT_PATH, exist_ok=True)

EXERCISE_STEPS_TEMPLATE = [
    {'id': 0, 'name_en': 'Rest', 'name_pt': 'Descansar', 'video': 'rest.mp4',
     'movement_type_en': 'Rest', 'movement_type_pt': 'Repouso', 'expected_movement': 'Rest'},
    {'id': 1, 'name_en': 'Reach for Cup', 'name_pt': 'Alcançar Copo', 'video': 'reach_cup.mp4',
     'movement_type_en': 'Supination', 'movement_type_pt': 'Supinação', 'expected_movement': 'Supination'},
    {'id': 2, 'name_en': 'Grasp Cup', 'name_pt': 'Agarrar Copo', 'video': 'grasp_cup.mp4',
     'movement_type_en': 'Grasp', 'movement_type_pt': 'Agarrar', 'expected_movement': 'Grasp'},
    {'id': 3, 'name_en': 'Lift Cup', 'name_pt': 'Levantar Copo', 'video': 'lift_cup.mp4',
     'movement_type_en': 'Flexion', 'movement_type_pt': 'Flexão', 'expected_movement': 'Flexion'},
    {'id': 4, 'name_en': 'Drink', 'name_pt': 'Beber', 'video': 'drink.mp4',
     'movement_type_en': 'Pronation', 'movement_type_pt': 'Pronação', 'expected_movement': 'Pronation'},
    {'id': 5, 'name_en': 'Lower Cup', 'name_pt': 'Pousar Copo', 'video': 'lower_cup.mp4',
     'movement_type_en': 'Extension', 'movement_type_pt': 'Extensão', 'expected_movement': 'Extension'},
]

SOUP_STEPS = [
    {**EXERCISE_STEPS_TEMPLATE[0]}, # Rest
    {'id': 6, 'name_en': 'Reach for Spoon', 'name_pt': 'Alcançar Colher', 'video': 'reach_spoon.mp4',
     'movement_type_en': 'Extension', 'movement_type_pt': 'Extensão', 'expected_movement': 'Extension'},
    {'id': 7, 'name_en': 'Grasp Spoon', 'name_pt': 'Agarrar Colher', 'video': 'grasp_spoon.mp4',
     'movement_type_en': 'Grasp', 'movement_type_pt': 'Agarrar', 'expected_movement': 'Grasp'},
    {'id': 8, 'name_en': 'Scoop Soup', 'name_pt': 'Apanhar Sopa', 'video': 'scoop_soup.mp4',
     'movement_type_en': 'Supination', 'movement_type_pt': 'Supinação', 'expected_movement': 'Supination'},
    {'id': 9, 'name_en': 'Bring Spoon to Mouth', 'name_pt': 'Levar Colher à Boca', 'video': 'bring_spoon_mouth.mp4',
     'movement_type_en': 'Flexion', 'movement_type_pt': 'Flexão', 'expected_movement': 'Flexion'},
    {'id': 10, 'name_en': 'Return Spoon to Bowl', 'name_pt': 'Devolver Colher à Tigela', 'video': 'return_spoon_bowl.mp4',
     'movement_type_en': 'Extension', 'movement_type_pt': 'Extensão', 'expected_movement': 'Extension'},
    {'id': 11, 'name_en': 'Lower Spoon', 'name_pt': 'Pousar Colher', 'video': 'lower_spoon.mp4',
     'movement_type_en': 'Pronation', 'movement_type_pt': 'Pronação', 'expected_movement': 'Pronation'},
]

BOOK_GRAB_STEPS = [
    {**EXERCISE_STEPS_TEMPLATE[0]}, # Rest
    {'id': 12, 'name_en': 'Reach for Book', 'name_pt': 'Alcançar Livro', 'video': 'reach_book.mp4',
     'movement_type_en': 'Extension', 'movement_type_pt': 'Extensão', 'expected_movement': 'Extension'},
    {'id': 13, 'name_en': 'Grasp Book', 'name_pt': 'Agarrar Livro', 'video': 'grasp_book.mp4',
     'movement_type_en': 'Grasp', 'movement_type_pt': 'Agarrar', 'expected_movement': 'Grasp'},
    {'id': 14, 'name_en': 'Lift Book', 'name_pt': 'Levantar Livro', 'video': 'lift_book.mp4',
     'movement_type_en': 'Flexion', 'movement_type_pt': 'Flexão', 'expected_movement': 'Flexion'},
    {'id': 15, 'name_en': 'Turn Book', 'name_pt': 'Roda Livro', 'video': 'turn_book_closer.mp4',
     'movement_type_en': 'Supination', 'movement_type_pt': 'Supinação', 'expected_movement': 'Supination'},
    {'id': 16, 'name_en': 'Lower Book', 'name_pt': 'Pousar Livro', 'video': 'lower_book.mp4',
     'movement_type_en': 'Extension', 'movement_type_pt': 'Extensão', 'expected_movement': 'Extension'},
]

DOOR_KNOB_STEPS = [
    {**EXERCISE_STEPS_TEMPLATE[0]}, # Rest
    {'id': 17, 'name_en': 'Reach for Door Knob', 'name_pt': 'Alcançar Maçaneta', 'video': 'reach_knob.mp4',
     'movement_type_en': 'Extension', 'movement_type_pt': 'Extensão', 'expected_movement': 'Extension'},
    {'id': 18, 'name_en': 'Grasp Door Knob', 'name_pt': 'Agarrar Maçaneta', 'video': 'grasp_knob.mp4',
     'movement_type_en': 'Grasp', 'movement_type_pt': 'Agarrar', 'expected_movement': 'Grasp'},
    {'id': 19, 'name_en': 'Turn Door Knob', 'name_pt': 'Rodar Maçaneta', 'video': 'turn_knob.mp4',
     'movement_type_en': 'Supination', 'movement_type_pt': 'Supinação', 'expected_movement': 'Supination'},
    {'id': 20, 'name_en': 'Retract Hand', 'name_pt': 'Recuar Mão', 'video': 'retract_hand_knob.mp4',
     'movement_type_en': 'Flexion', 'movement_type_pt': 'Flexão', 'expected_movement': 'Flexion'},
    {'id': 21, 'name_en': 'Release Door Knob', 'name_pt': 'Largar Maçaneta', 'video': 'release_knob.mp4',
     'movement_type_en': 'Rest', 'movement_type_pt': 'Repouso', 'expected_movement': 'Rest'},
]
# --- Text Strings (internationalization) ---
STRINGS = {
    'en': {
        'window_title': "Stroke Rehabilitation Assistant",
        'feedback_initializing': "Initializing...",
        'feedback_no_movement': "No movement detected. Please start.",
        'feedback_incorrect': "Incorrect Movement Detected. Please try again.",
        'feedback_weak': "Correct Movement, but more force needed.",
        'feedback_strong': "Correct Movement + Sufficient Force! Well done!",
        'feedback_next_step': "Moving to the next step...",
        'feedback_finished': "Exercise sequence complete! Well Done!",
        'button_next_manual': "Next Step (Manual)",
        'button_change_lang': "Mudar para Português",
        'button_change_lang_dialog': "Mudar para Português",
        'current_step': "Current Step:",
        'instruction': "Follow the video.",
        'status_arduino_connected': "Arduino Connected",
        'status_arduino_disconnected': "Arduino Disconnected",
        'status_arduino_error': "Arduino Comms Error",
        'final_image_file': 'finished_en.png',
        'button_play_again': "Play Again",
        'button_exit': "Exit",
        'button_help': "Help",
        'button_open_report_folder': "Open Report Folder",
        'help_message_title': "Help / Instructions",
        'help_message_text': (
            "Welcome to the Stroke Rehabilitation Assistant!\n\n"
            "1. Select or enter patient name.\n"
            "2. Select an exercise sequence.\n"
            "3. Follow the video instructions for each step.\n"
            "4. Feedback is shown by the colored bar.\n"
            "5. After finishing, you can 'Play Again', 'Create Summary', or 'Exit'.\n"
            "6. Session reports are saved in 'output/patient_name/'."
        ),
        'report_title': "Rehabilitation Session Report", 'report_datetime': "Date & Time",
        'report_patient_header': "Patient",
        'report_sequence_name': "Exercise Sequence", 'report_total_duration': "Total Duration",
        'report_step_details_header': "Step Details", 'report_table_header_step_num': "Step #",
        'report_table_header_name': "Name",
        'report_table_header_movement_type': "Movement Type", 
        'report_table_header_time': "Time Taken (s)",
        'report_table_header_incorrect': "Incorrect", 'report_table_header_weak': "Weak",
        'report_table_header_no_movement': "No Movement", 'report_table_header_manual_advance': "Manual Advance",
        'report_value_yes': "Yes", 'report_value_no': "No", 'report_value_na': "N/A",
        'patient_dialog_title': "Select Patient",
        'patient_dialog_new_label': "Enter New Patient Name:",
        'patient_dialog_existing_label': "Or Select Existing Patient:",
        'patient_dialog_continue_button': "Continue to Exercises",
        'patient_dialog_no_name_error_title': "Name Required",
        'patient_dialog_no_name_error_text': "Please enter a new patient name or select an existing one.",
        'patient_dialog_select_patient_for_folder': "Select or enter a patient to open their report folder.",
        'patient_dialog_patient_folder_not_exist': "Report folder for this patient does not exist yet. It will be created when a session is saved.",
        'exercise_selection_dialog_title': "Select Exercise Sequence",
        'exercise_selection_dialog_title_for_patient': "Select Exercise for {patient_name}",
        # Translatable sequence names
        'sequence_cup_name': "Cup Sequence",
        'sequence_rest_only_name': "Rest Only",
        'sequence_short_name': "Short Sequence",
        'sequence_soup_name': "Soup Sequence",
        'sequence_grab_book_name': "Book Grab Sequence",
        'sequence_turn_door_knob_name': "Open Door Sequence",
        # Summary Report
        'button_create_summary': "Create Summary",
        'summary_report_title': "Patient Summary Report",
        'report_generated_on': "Report generated on",
        'summary_overview_header': "Overall Summary",
        'summary_total_sessions': "Total Sessions",
        'summary_total_train_time': "Total Training Time",
        'summary_mean_session_time': "Mean Session Time",
        'summary_overall_step_success_rate': "Overall Step Success Rate (Perfect First Try)",
        'summary_overall_step_no_incorrect_rate': "Overall Step Completion Rate (No Incorrect Attempts)",
        'summary_most_problematic_step': "Most Problematic Step (by total errors)",
        'summary_most_problematic_sequence': "Most Problematic Sequence (by total errors)",
        'summary_movement_type_header': "Performance by Movement Type",
        'summary_sequence_performance_header': "Performance by Exercise Sequence",
        'summary_times_performed': "Times Performed",
        'graph_xlabel_date': "Date",
        'graph_ylabel_sessions': "Number of Sessions",
        'graph_title_sessions_per_day': "Training Sessions per Day",
        'summary_mean_duration_this_sequence': "Mean Duration for this Sequence",
        'summary_avg_time_per_step': "Avg. Time (s)",
        'summary_total_incorrect': "Total Incorrect",
        'summary_total_weak': "Total Weak",
        'summary_total_no_movement': "Total No Movement",
        'summary_total_manual_advance': "Total Manual Adv.",
        'summary_step_perfection_rate': "Perfection Rate (%)",
    },
    'pt': {
        'window_title': "Assistente de Reabilitação AVC",
        'feedback_initializing': "A inicializar...",
        'feedback_no_movement': "Nenhum movimento detetado. Por favor, comece.",
        'feedback_incorrect': "Movimento Incorreto Detetado. Tente novamente.",
        'feedback_weak': "Movimento Correto, mas precisa de mais força.",
        'feedback_strong': "Movimento Correto + Força Suficiente! Bom trabalho!",
        'feedback_next_step': "A avançar para o próximo passo...",
        'feedback_finished': "Sequência de exercícios completa! Parabéns!",
        'button_next_manual': "Próximo Passo (Manual)",
        'button_change_lang': "Switch to English",
        'button_change_lang_dialog': "Switch to English",
        'current_step': "Passo Atual:",
        'instruction': "Siga o vídeo.",
        'status_arduino_connected': "Arduino Ligado", 'status_arduino_disconnected': "Arduino Desligado",
        'status_arduino_error': "Erro Comunicação Arduino", 'final_image_file': 'finished_pt.png',
        'button_play_again': "Jogar Novamente", 'button_exit': "Sair", 'button_help': "Ajuda",
        'button_open_report_folder': "Abrir Pasta Relatórios",
        'help_message_title': "Ajuda / Instruções",
        'help_message_text': (
            "Bem-vindo ao Assistente de Reabilitação AVC!\n\n"
            "1. Selecione ou introduza o nome do paciente.\n"
            "2. Selecione uma sequência de exercícios.\n"
            "3. Siga as instruções do vídeo para cada passo.\n"
            "4. O feedback é mostrado na barra colorida.\n"
            "5. Após terminar, pode 'Jogar Novamente', 'Criar Resumo' ou 'Sair'.\n"
            "6. Os relatórios das sessões são guardados em 'output/nome_do_paciente/'."
        ),
        'report_title': "Relatório da Sessão de Reabilitação", 'report_datetime': "Data e Hora",
        'report_patient_header': "Paciente",
        'report_sequence_name': "Sequência de Exercício", 'report_total_duration': "Duração Total",
        'report_step_details_header': "Detalhes dos Passos", 'report_table_header_step_num': "Passo N.º",
        'report_table_header_name': "Nome",
        'report_table_header_movement_type': "Tipo de Movimento",
        'report_table_header_time': "Tempo Gasto (s)",
        'report_table_header_incorrect': "Incorretas", 'report_table_header_weak': "Fracas",
        'report_table_header_no_movement': "Sem Movimento", 'report_table_header_manual_advance': "Avanço Manual",
        'report_value_yes': "Sim", 'report_value_no': "Não", 'report_value_na': "N/D",
        'patient_dialog_title': "Selecionar Paciente",
        'patient_dialog_new_label': "Introduza Nome do Novo Paciente:",
        'patient_dialog_existing_label': "Ou Selecione Paciente Existente:",
        'patient_dialog_continue_button': "Continuar para Exercícios",
        'patient_dialog_no_name_error_title': "Nome Necessário",
        'patient_dialog_no_name_error_text': "Por favor, introduza um nome para novo paciente ou selecione um existente.",
        'patient_dialog_select_patient_for_folder': "Selecione ou introduza um paciente para abrir a sua pasta de relatórios.",
        'patient_dialog_patient_folder_not_exist': "A pasta de relatórios para este paciente ainda não existe. Será criada ao guardar uma sessão.",
        'exercise_selection_dialog_title': "Selecionar Sequência de Exercício",
        'exercise_selection_dialog_title_for_patient': "Selecionar Exercício para {patient_name}",
        # Translatable sequence names
        'sequence_cup_name': "Sequência do Copo",
        'sequence_rest_only_name': "Apenas Descanso",
        'sequence_short_name': "Sequência Curta",
        'sequence_soup_name': "Sequência da Sopa",
        'sequence_grab_book_name': "Sequência Alcançar Livro",
        'sequence_turn_door_knob_name': "Sequência Abrir Porta",
        # Summary Report
        'button_create_summary': "Criar Resumo",
        'summary_report_title': "Relatório Resumo do Paciente",
        'report_generated_on': "Relatório gerado em",
        'summary_overview_header': "Resumo Geral",
        'summary_total_sessions': "Total de Sessões",
        'summary_total_train_time': "Tempo Total de Treino",
        'summary_mean_session_time': "Tempo Médio por Sessão",
        'summary_overall_step_success_rate': "Taxa de Sucesso Geral por Passo (Perfeito à Primeira)",
        'summary_overall_step_no_incorrect_rate': "Taxa de Conclusão de Passo (Sem Tentativas Incorretas)",
        'summary_most_problematic_step': "Passo Mais Problemático (por total de erros)",
        'summary_most_problematic_sequence': "Sequência Mais problemática (por total de erros)",
        'summary_movement_type_header': "Desempenho por Tipo de Movimento",
        'summary_sequence_performance_header': "Desempenho por Sequência de Exercício",
        'summary_times_performed': "Vezes Realizado",
        'graph_xlabel_date': "Data",
        'graph_ylabel_sessions': "Número de Sessões",
        'graph_title_sessions_per_day': "Sessões de Treino por Dia",
        'summary_mean_duration_this_sequence': "Duração Média para esta Sequência",
        'summary_avg_time_per_step': "Tempo Médio (s)",
        'summary_total_incorrect': "Total Incorretas",
        'summary_total_weak': "Total Fracas",
        'summary_total_no_movement': "Total Sem Movimento",
        'summary_total_manual_advance': "Total Avanço Manual",
        'summary_step_perfection_rate': "Taxa de Perfeição (%)",
    }
}

# --- Sound Map ---
SOUND_MAP = {
    'NO_MOVEMENT': None, 'INCORRECT_MOVEMENT': SOUND_PATH + 'fail.wav',
    'CORRECT_WEAK': SOUND_PATH + 'weak.wav', 'CORRECT_STRONG': SOUND_PATH + 'success.wav',
    'NEXT_STEP': SOUND_PATH + 'next_step.wav', 'FINISHED': SOUND_PATH + 'finished.wav'
}

# --- Exercise Sequences Definition ---
EXERCISE_SEQUENCES = {
    # Internal Key : (Translatable String Key for Name, Steps List)
    'Rest Only': ('sequence_rest_only_name', [EXERCISE_STEPS_TEMPLATE[0]]),
    'Cup Sequence': ('sequence_cup_name', EXERCISE_STEPS_TEMPLATE),
    # 'Short Sequence': ('sequence_short_name', EXERCISE_STEPS_TEMPLATE[:3]),
    'Soup Sequence': ('sequence_soup_name', SOUP_STEPS),
    'Book Grab Sequence': ('sequence_grab_book_name', BOOK_GRAB_STEPS),
    'Turn Door Knob Sequence': ('sequence_turn_door_knob_name', DOOR_KNOB_STEPS) 
}

class MovementTracker:
    """Tracks completed movements for exercise sequences"""
    def __init__(self, confidence_threshold=0.6, min_duration_s=0.3):
        self.confidence_threshold = confidence_threshold
        self.min_duration_s = min_duration_s
        
        self.current_movement = "Rest"
        self.movement_start_time = None
        self.completed_movements = []
        self.movement_in_progress = False
        
    def update(self, movement, confidence, timestamp):
        """Update movement state and detect completed movements"""
        completed_movement = None
        #print(f"Tracker update: {movement}, conf={confidence:.2f}")
        # Check for state changes
        if movement != self.current_movement:
            # Transitioning to a new movement
            if movement != "Rest" and confidence >= self.confidence_threshold:
                # Starting a movement
                if not self.movement_in_progress:
                    print(f"[{timestamp:.2f}] Starting movement: {movement} ({confidence:.2f})")
                    self.movement_start_time = timestamp
                    self.movement_in_progress = True
                else:
                    # Changing movement type mid-movement
                    print(f"[{timestamp:.2f}] Movement changed: {self.current_movement} → {movement}")
                    
                    # Complete the previous movement if it was long enough
                    if self.movement_start_time is not None:
                        duration = timestamp - self.movement_start_time
                        if duration >= self.min_duration_s:
                            completed_movement = {
                                'type': self.current_movement,
                                'start_time': self.movement_start_time,
                                'end_time': timestamp,
                                'duration': duration,
                                'confidence': confidence
                            }
                            self.completed_movements.append(completed_movement)
                            print(f"[{timestamp:.2f}] Completed {self.current_movement} ({duration:.2f}s)")
                    
                    # Start tracking the new movement
                    self.movement_start_time = timestamp
            
            elif movement == "Rest" and self.movement_in_progress:
                # Completing a movement
                if self.movement_start_time is not None:
                    duration = timestamp - self.movement_start_time
                    if duration >= self.min_duration_s:
                        completed_movement = {
                            'type': self.current_movement,
                            'start_time': self.movement_start_time,
                            'end_time': timestamp,
                            'duration': duration,
                            'confidence': confidence
                        }
                        self.completed_movements.append(completed_movement)
                        print(f"[{timestamp:.2f}] Completed {self.current_movement} ({duration:.2f}s)")
                
                self.movement_in_progress = False
                self.movement_start_time = None
        
        # Update current movement
        self.current_movement = movement
        
        return completed_movement


CURRENT_EXERCISE_STEPS = []

# --- Patient Dialog ---
class PatientDialog(QDialog): # Keep as is (already refactored)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_patient_name = None
        if hasattr(QApplication.instance(), 'current_app_language'):
            self.dialog_language = QApplication.instance().current_app_language
        else:
            self.dialog_language = DEFAULT_LANGUAGE
        self.main_layout = QVBoxLayout(self)
        self.new_patient_label = QLabel()
        self.patient_name_input = QLineEdit()
        self.existing_patient_label = QLabel()
        self.existing_patients_combo = QComboBox()
        self.lang_button_dialog = QPushButton()
        self.open_report_folder_button = QPushButton()
        self.continue_button = QPushButton()
        self.main_layout.addWidget(self.new_patient_label)
        self.main_layout.addWidget(self.patient_name_input)
        self.main_layout.addWidget(self.existing_patient_label)
        self.main_layout.addWidget(self.existing_patients_combo)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.lang_button_dialog)
        button_layout.addWidget(self.open_report_folder_button)
        button_layout.addStretch()
        button_layout.addWidget(self.continue_button)
        self.main_layout.addLayout(button_layout)
        self.patient_name_input.textChanged.connect(lambda text: self.existing_patients_combo.setCurrentIndex(0) if text else None)
        self.existing_patients_combo.currentIndexChanged.connect(lambda index: self.patient_name_input.clear() if index > 0 else None)
        self.lang_button_dialog.clicked.connect(self._toggle_dialog_language)
        self.open_report_folder_button.clicked.connect(self._open_report_folder_action)
        self.continue_button.clicked.connect(self._accept_selection)
        self._update_dialog_texts(); self._populate_existing_patients(); self.setMinimumSize(350, 250)
    def _open_report_folder_action(self):
        name_to_open = self.patient_name_input.text().strip()
        if not name_to_open and self.existing_patients_combo.currentText():
            name_to_open = self.existing_patients_combo.currentText()
        if not name_to_open:
            QMessageBox.information(self, STRINGS[self.dialog_language].get('button_open_report_folder'),
                                    STRINGS[self.dialog_language].get('patient_dialog_select_patient_for_folder'))
            return
        safe_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in name_to_open)
        if not safe_name: # Should not happen if button is enabled correctly
             QMessageBox.warning(self, "Invalid Name", "Patient name results in an invalid folder name.")
             return
        patient_folder_path = os.path.join(OUTPUT_PATH, safe_name)
        if not os.path.isdir(patient_folder_path):
            # Optionally create it, or inform user it will be created.
            # For now, just inform, as it's created during report saving.
            QMessageBox.information(self, STRINGS[self.dialog_language].get('button_open_report_folder'),
                                    STRINGS[self.dialog_language].get('patient_dialog_patient_folder_not_exist'))
            # We can still try to open the parent OUTPUT_PATH if patient folder doesn't exist
            # Or, just do nothing. For now, let's try to open it, or its parent if it doesn't exist.
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(patient_folder_path)):
                # If opening specific patient folder fails (e.g. doesn't exist), try opening base OUTPUT_PATH
                if not QDesktopServices.openUrl(QUrl.fromLocalFile(OUTPUT_PATH)):
                    QMessageBox.warning(self, "Error", f"Could not open folder: {OUTPUT_PATH}")
        else:
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(patient_folder_path)):
                QMessageBox.warning(self, "Error", f"Could not open folder: {patient_folder_path}")
    def _update_dialog_texts(self):
        self.setWindowTitle(STRINGS[self.dialog_language].get('patient_dialog_title', "Select Patient"))
        self.new_patient_label.setText(STRINGS[self.dialog_language].get('patient_dialog_new_label', "Enter New Patient Name:"))
        self.patient_name_input.setPlaceholderText("e.g., John_Doe")
        self.existing_patient_label.setText(STRINGS[self.dialog_language].get('patient_dialog_existing_label', "Or Select Existing Patient:"))
        self.lang_button_dialog.setText(STRINGS[self.dialog_language].get('button_change_lang_dialog', "Change Language"))
        self.continue_button.setText(STRINGS[self.dialog_language].get('patient_dialog_continue_button', "Continue"))
        self.lang_button_dialog.adjustSize(); self.lang_button_dialog.update()
        self.open_report_folder_button.setText(STRINGS[self.dialog_language].get('button_open_report_folder'))
        self.continue_button.adjustSize(); self.continue_button.update()
    def _populate_existing_patients(self):
        current_selection = self.existing_patients_combo.currentText()
        self.existing_patients_combo.clear(); self.existing_patients_combo.addItem("")
        if not os.path.isdir(OUTPUT_PATH): return
        try:
            patients = sorted([item for item in os.listdir(OUTPUT_PATH) if os.path.isdir(os.path.join(OUTPUT_PATH, item))])
            for patient in patients: self.existing_patients_combo.addItem(patient)
            index = self.existing_patients_combo.findText(current_selection)
            if index != -1: self.existing_patients_combo.setCurrentIndex(index)
        except OSError as e: print(f"Error scanning for existing patients: {e}")
    def _accept_selection(self):
        name_input = self.patient_name_input.text().strip(); name_combo = self.existing_patients_combo.currentText()
        if name_input:
            self.selected_patient_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in name_input)
            if not self.selected_patient_name:
                 QMessageBox.warning(self, STRINGS[self.dialog_language].get('patient_dialog_no_name_error_title', "Name Required"), STRINGS[self.dialog_language].get('patient_dialog_no_name_error_text', "Invalid chars.")); self.selected_patient_name = None; return
        elif name_combo: self.selected_patient_name = name_combo
        else: QMessageBox.warning(self, STRINGS[self.dialog_language].get('patient_dialog_no_name_error_title', "Name Required"), STRINGS[self.dialog_language].get('patient_dialog_no_name_error_text', "Please enter or select.")); return
        print(f"Patient selected/entered: {self.selected_patient_name}"); self.accept()
    def _toggle_dialog_language(self):
        current_idx = LANGUAGES.index(self.dialog_language); next_idx = (current_idx + 1) % len(LANGUAGES)
        self.dialog_language = LANGUAGES[next_idx]
        if hasattr(QApplication.instance(), 'current_app_language'): QApplication.instance().current_app_language = self.dialog_language
        print(f"PatientDialog language changed to: {self.dialog_language}"); self._update_dialog_texts(); self._populate_existing_patients()

# --- Selection Dialog (for Exercises) ---
class SelectionDialog(QDialog): # MODIFIED
    def __init__(self, patient_name, parent=None):
        super().__init__(parent)
        self.patient_name_for_title = patient_name
        self.selected_sequence_name_internal = None # Store the English key
        self.selected_sequence_steps = []

        if hasattr(QApplication.instance(), 'current_app_language'):
            self.dialog_language = QApplication.instance().current_app_language
        else:
            self.dialog_language = DEFAULT_LANGUAGE

        self.main_layout = QVBoxLayout(self)
        
        # Create widgets ONCE
        self.exercise_buttons = [] # List to hold exercise QPushButtons
        for _ in EXERCISE_SEQUENCES: # Create enough buttons
            btn = QPushButton()
            self.exercise_buttons.append(btn)
            self.main_layout.addWidget(btn)
            
        self.lang_button_dialog = QPushButton()
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.lang_button_dialog)
        self.main_layout.addLayout(button_layout)

        # Connect signals for language button
        self.lang_button_dialog.clicked.connect(self._toggle_dialog_language)
        
        self._update_dialog_texts() # Initial text setup and button connections
        self.resize(400, 300)

    def _update_dialog_texts(self):
        """Updates all translatable texts and reconfigures exercise buttons."""
        if self.patient_name_for_title:
            title = STRINGS[self.dialog_language].get('exercise_selection_dialog_title_for_patient', "Select Exercise for {patient_name}").format(patient_name=self.patient_name_for_title)
        else:
            title = STRINGS[self.dialog_language].get('exercise_selection_dialog_title', "Select Exercise Sequence")
        self.setWindowTitle(title)

        # Update exercise buttons
        for i, (internal_key, (trans_key, steps_list)) in enumerate(EXERCISE_SEQUENCES.items()):
            if i < len(self.exercise_buttons):
                btn = self.exercise_buttons[i]
                translated_name = STRINGS[self.dialog_language].get(trans_key, internal_key) # Fallback to internal key
                btn.setText(translated_name)
                # Disconnect old lambda before connecting new one to avoid multiple calls
                try: btn.clicked.disconnect() 
                except RuntimeError: pass # No connection to disconnect
                btn.clicked.connect(lambda checked=False, ik=internal_key, tk=trans_key, s=steps_list: self.select(ik, tk, s))
                btn.adjustSize(); btn.update()

        self.lang_button_dialog.setText(STRINGS[self.dialog_language].get('button_change_lang_dialog', "Change Language"))
        self.lang_button_dialog.adjustSize()
        self.lang_button_dialog.update()

    # Modified select to store internal key and use translated key for display name if needed later
    def select(self, internal_key, trans_key_for_name, steps):
        global CURRENT_EXERCISE_STEPS
        self.selected_sequence_name_internal = internal_key # Store the English key
        # For reporting or display, we might want the translated name.
        # For now, MainWindow uses self.current_sequence_name which is the internal key.
        # If MainWindow needs the translated name, it can get it via self.tr(trans_key_for_name)
        self.selected_sequence_name = STRINGS[self.dialog_language].get(trans_key_for_name, internal_key) # Store translated name for consistency if needed
        
        self.selected_sequence_steps = steps
        CURRENT_EXERCISE_STEPS = steps
        print(f"Selected sequence (internal key): {self.selected_sequence_name_internal} -> Display: {self.selected_sequence_name} ({len(self.selected_sequence_steps)} steps)")
        self.accept()

    def _toggle_dialog_language(self):
        current_idx = LANGUAGES.index(self.dialog_language); next_idx = (current_idx + 1) % len(LANGUAGES)
        self.dialog_language = LANGUAGES[next_idx]
        if hasattr(QApplication.instance(), 'current_app_language'): QApplication.instance().current_app_language = self.dialog_language
        print(f"SelectionDialog language changed to: {self.dialog_language}")
        self._update_dialog_texts()

# --- Worker Thread ---
class PDFGenerationWorker(QObject):
    # Signal to indicate completion, carrying the status message
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, patient_name, base_output_path, app_strings_dict, app_exercise_sequences_def, attempt_pdf):
        super().__init__()
        self.patient_name = patient_name
        self.base_output_path = base_output_path
        self.app_strings_dict = app_strings_dict
        self.app_exercise_sequences_def = app_exercise_sequences_def
        self.attempt_pdf = attempt_pdf

    @Slot()
    def process_report_generation(self):
        """This slot will run in the separate thread."""
        try:
            print("PDFGenerationWorker: Starting summary report generation...")
            status_message = generate_summary_report_for_patient(
                patient_name=self.patient_name,
                base_output_path=self.base_output_path,
                app_strings_dict=self.app_strings_dict,
                app_exercise_sequences_def=self.app_exercise_sequences_def,
                attempt_pdf_generation=self.attempt_pdf
            )
            self.finished.emit(status_message)
            print("PDFGenerationWorker: Finished summary report generation.")
        except Exception as e:
            error_msg = f"Error during PDF generation: {e}"
            print(f"PDFGenerationWorker: {error_msg}")
            self.error.emit(error_msg)

# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self, patient_name, selected_sequence_name, selected_steps):
        super().__init__()
        if hasattr(QApplication.instance(), 'current_app_language'): 
            self.current_language = QApplication.instance().current_app_language
        else: 
            self.current_language = DEFAULT_LANGUAGE
            
        self.current_step_index = -1
        self.arduino = None
        self.arduino_status = "disconnected"
        self.last_successful_status_time = 0
        self.advance_on_success = True
        self.close_for_restart_flag = False
        
        self.patient_name = patient_name
        self.session_start_time = None
        self.current_sequence_name = selected_sequence_name
        self.current_exercise_steps_definition = selected_steps
        self.session_report_data = []
        self.current_step_start_time = None
        self.current_step_attempts = {}
        
        # Initialize threading references
        self.daq_thread = None
        self.processing_thread = None
        self.daq_worker = None
        self.processing_worker = None
        self.emg_running = False
        
        self._start_new_session_tracking()
        
        self.sound_effects = {}
        self.preload_sounds()
        
        # Set up UI first
        self.setup_ui()
        self.setWindowIcon(QIcon(IMAGE_PATH + "app_icon.png"))
        self.media_player.mediaStatusChanged.connect(self.handle_media_status_changed)
        
        # Set up Arduino
        self.setup_arduino()
        
        # Create movement tracker
        self.movement_tracker = MovementTracker(
            confidence_threshold=0.6,
            min_duration_s=0.3
        )
        
        # Set up EMG controls (UI components)
        self.setup_emg_controls()
        
        # PDF thread references
        self.pdf_thread = None
        self.pdf_worker = None
        
        # Start the first step
        self.advance_step()
        
        # IMPORTANT: Set up EMG threading AFTER everything else is initialized
        # And do it with a slight delay to ensure UI is fully ready
        QTimer.singleShot(1000, self.initialize_emg_system)
    
    def initialize_emg_system(self):
        """Initialize EMG system after UI is fully ready"""
        try:
            print("Initializing EMG system...")
            self.setup_emg_threading()
            print("EMG system initialized successfully")
        except Exception as e:
            print(f"Error initializing EMG system: {e}")
            # Don't crash the app if EMG fails to initialize
            if hasattr(self, 'emg_status_label'):
                self.emg_status_label.setText(f"EMG Status: Failed to initialize - {e}")

    def _start_new_session_tracking(self): # Keep as is
        self.session_start_time = datetime.datetime.now(); self.session_report_data = []
        # For display/report, get translated sequence name
        sequence_display_name = self.current_sequence_name # Default to internal key
        for internal_key, (trans_key, _) in EXERCISE_SEQUENCES.items():
            if internal_key == self.current_sequence_name:
                sequence_display_name = self.tr(trans_key)
                break
        print(f"Starting new session tracking for patient '{self.patient_name}', sequence '{sequence_display_name}' at {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")


    def tr(self, text_key): # Keep as is
        return STRINGS[self.current_language].get(text_key, f"<{text_key}>")

    def preload_sounds(self): # Keep as is
        print("Preloading sounds...")
        self.sound_effects = {}
        print(f"  -> Initial self.sound_effects: {list(self.sound_effects.keys())}")
        for key, filename in SOUND_MAP.items():
            print(f"  --> Processing Key: '{key}', Filename: '{filename}'")
            if filename:
                url = QUrl.fromLocalFile(filename)
                if not url.isValid() or url.isEmpty(): print(f"  -> Warning: Sound file invalid or not found: {filename}. Skipping key '{key}'."); continue
                sound = QSoundEffect(self); sound.setSource(url); sound.setVolume(1.0)
                current_status = sound.status()
                status_text = { QSoundEffect.Null: "Null", QSoundEffect.Loading: "Loading", QSoundEffect.Ready: "Ready", QSoundEffect.Error: "Error" }.get(current_status, "Unknown")
                print(f"  -> Status for '{filename}' after setSource: {status_text} ({current_status})")
                if current_status == QSoundEffect.Error: print(f"  -> Error confirmed for '{key}', NOT storing.")
                else: self.sound_effects[key] = sound; print(f"  -> Stored sound effect for key: '{key}'")
            else: self.sound_effects[key] = None; print(f"  -> Stored None for key: '{key}'")
        print(f"--- Finished preloading. FINAL keys in self.sound_effects: {list(self.sound_effects.keys())} ---")

    def setup_ui(self):
        """Sets up the main UI components of the application window."""
        # --- Basic Window Setup ---
        self.setWindowTitle(self.tr('window_title'))
        self.setGeometry(100, 100, 1000, 700) # Initial window size and position

        # --- Central Widget and Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget) # Main vertical layout

        # --- Stacked Widget (for switching between Exercise/Finished screens) ---
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget, stretch=4) # Give stacked widget more vertical space

        # --- Exercise Screen Setup ---
        self.exercise_widget = QWidget() # Container for exercise screen elements
        exercise_layout = QVBoxLayout(self.exercise_widget)

        # Top part: Video + Plot
        ex_top_layout = QHBoxLayout()
        exercise_layout.addLayout(ex_top_layout, stretch=3)

        # Video Player
        self.video_widget = QVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        ex_top_layout.addWidget(self.video_widget, stretch=3) # Video takes more horizontal space

        # EMG Plot
        self.plot_widget = pg.PlotWidget(title="EMG Signal (Simulated)")
        self.plot_widget.setYRange(-500, 500)
        self.plot_widget.showGrid(x=False, y=True)
        self.emg_curve = self.plot_widget.plot(pen='b') # EMG plot line
        ex_top_layout.addWidget(self.plot_widget, stretch=2)

        # Info part: Step Label + Instructions
        ex_info_layout = QHBoxLayout()
        self.step_label = QLabel(f"{self.tr('current_step')} -") # Current step label
        self.instruction_label = QLabel(self.tr('instruction')) # Instruction label
        ex_info_layout.addWidget(self.step_label, stretch=1)
        ex_info_layout.addWidget(self.instruction_label, stretch=2)
        exercise_layout.addLayout(ex_info_layout)

        self.stacked_widget.addWidget(self.exercise_widget) # Add exercise screen to the stack

        # --- Finished Screen Setup ---
        self.finished_widget = QWidget() # Container for finished screen elements
        # Set background to white for PNG transparency
        self.finished_widget.setStyleSheet("background-color: white;")
        finished_layout = QVBoxLayout(self.finished_widget) # Vertical layout for centering

        # Finished Image Label
        self.finished_image_label = QLabel()
        self.finished_image_label.setAlignment(Qt.AlignCenter) # Center pixmap within label
        # Size policy allows label to expand
        self.finished_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Manual scaling with SmoothTransformation is done in load_step

        # Finished Screen Buttons (horizontally centered)
        finished_buttons_layout = QHBoxLayout()
        finished_buttons_layout.addStretch() # Left stretch
        self.play_again_button = QPushButton(self.tr('button_play_again'))
        self.play_again_button.clicked.connect(self.restart_exercise)
        self.create_summary_button = QPushButton(self.tr('button_create_summary'))
        self.create_summary_button.clicked.connect(self._create_summary_report)
        self.open_report_folder_button_finished = QPushButton(self.tr('button_open_report_folder'))
        self.open_report_folder_button_finished.clicked.connect(self._open_current_patient_report_folder)
        self.exit_button = QPushButton(self.tr('button_exit'))
        self.exit_button.clicked.connect(self.close_application)
        finished_buttons_layout.addWidget(self.play_again_button)
        finished_buttons_layout.addWidget(self.create_summary_button)
        finished_buttons_layout.addWidget(self.open_report_folder_button_finished)
        finished_buttons_layout.addWidget(self.exit_button)
        finished_buttons_layout.addStretch() # Right stretch

        # Add elements to finished_layout for vertical centering
        finished_layout.addStretch(1) # Top stretch (less priority)
        # Give image label high stretch factor for maximum vertical space
        finished_layout.addWidget(self.finished_image_label, stretch=5)
        # Add buttons layout with no stretch factor (minimum vertical space)
        finished_layout.addLayout(finished_buttons_layout, stretch=0)
        finished_layout.addStretch(1) # Bottom stretch (less priority)

        self.stacked_widget.addWidget(self.finished_widget) # Add finished screen to the stack
        # --- End of Finished Screen Setup ---

        # --- Bottom Controls Area ---
        bottom_controls_layout = QHBoxLayout()
        main_layout.addLayout(bottom_controls_layout, stretch=1) # Give less vertical space to bottom controls

        # Feedback Label (colored bar)
        self.feedback_label = QLabel(self.tr('feedback_initializing'))
        self.feedback_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) # Expand horizontally
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setAutoFillBackground(True) # Needed for background color
        self.set_feedback_style('initializing')
        bottom_controls_layout.addWidget(self.feedback_label, stretch=3) # Feedback takes more space

        # Main Buttons Widget (Next, Lang, Help)
        self.main_button_widget = QWidget()
        main_button_layout = QVBoxLayout(self.main_button_widget)
        main_button_layout.setContentsMargins(0,0,0,0) # No margins
        self.manual_next_button = QPushButton(self.tr('button_next_manual'))
        self.manual_next_button.clicked.connect(self.manual_advance_step)
        self.lang_button = QPushButton(self.tr('button_change_lang'))
        self.lang_button.clicked.connect(self.toggle_language)
        self.help_button = QPushButton(self.tr('button_help'))
        self.help_button.clicked.connect(self.show_help)
        main_button_layout.addWidget(self.manual_next_button)
        main_button_layout.addWidget(self.lang_button)
        main_button_layout.addWidget(self.help_button)
        bottom_controls_layout.addWidget(self.main_button_widget, stretch=1)

        # Arduino Status Label
        self.arduino_status_label = QLabel(f"Arduino: {self.tr('status_arduino_disconnected')}")
        self.arduino_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter) # Align right
        bottom_controls_layout.addWidget(self.arduino_status_label, stretch=1)

        # --- Initial State ---
        self.stacked_widget.setCurrentWidget(self.exercise_widget) # Show exercise screen first

    def _open_current_patient_report_folder(self):
        """Opens the report folder for the current patient."""
        if not self.patient_name:
            print("No patient name available to open report folder.")
            return
        patient_folder_path = os.path.join(OUTPUT_PATH, self.patient_name)
        # Ensure the base patient folder exists, QDesktopServices might not open non-existent paths well.
        os.makedirs(patient_folder_path, exist_ok=True) 
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(patient_folder_path)):
            QMessageBox.warning(self, "Error", f"Could not open folder: {patient_folder_path}")

    def setup_arduino(self): # Keep as is
        if SERIAL_PORT:
            try: self.arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1); time.sleep(2); self.arduino_status = "connected"; print(f"Successfully connected to Arduino on {SERIAL_PORT}")
            except serial.SerialException as e: self.arduino = None; self.arduino_status = "error"; print(f"Error connecting to Arduino on {SERIAL_PORT}: {e}")
        else: self.arduino_status = "disconnected"; print("Serial port not configured. Arduino control disabled.")
        self.update_arduino_status_label()

    def update_arduino_status_label(self): # Keep as is (already calls adjustSize/update)
        status_key = f"status_arduino_{self.arduino_status}"; text = f"Arduino: {self.tr(status_key)}"; style = "color: gray;"
        if self.arduino_status == "connected": style = "color: green;"
        elif self.arduino_status == "error": style = "color: red;"
        self.arduino_status_label.setText(text); self.arduino_status_label.setStyleSheet(style)
        self.arduino_status_label.adjustSize(); self.arduino_status_label.update()

    def load_step(self): # Keep as is
        if 0 <= self.current_step_index < len(self.current_exercise_steps_definition):
            self.stacked_widget.setCurrentWidget(self.exercise_widget); self.main_button_widget.show()
            step_info = self.current_exercise_steps_definition[self.current_step_index]
            step_name = step_info[f'name_{self.current_language}']
            video_file = step_info['video']
            self.step_label.setText(self.tr('current_step') + f" {self.current_step_index + 1}/{len(self.current_exercise_steps_definition)}: {step_name}")
            self.instruction_label.setText(self.tr('instruction'))
            video_url = QUrl.fromLocalFile(VIDEO_PATH + video_file)
            if video_url.isValid(): self.media_player.setSource(video_url); self.media_player.setPosition(0); self.media_player.play(); print(f"Playing video: {video_file}")
            else: print(f"Error: Video file not found or invalid: {VIDEO_PATH + video_file}"); self.feedback_label.setText(f"Error loading video: {video_file}"); self.set_feedback_style('error')
            self.feedback_label.setText(self.tr('feedback_no_movement')); self.set_feedback_style('no_movement')
            self.last_successful_status_time = 0
            self.current_step_start_time = time.time()
            self.current_step_attempts = {'INCORRECT_MOVEMENT': 0, 'CORRECT_WEAK': 0, 'NO_MOVEMENT': 0}
            print(f"Starting step {self.current_step_index + 1}: {step_name}")
        else:
            print("Exercise sequence finished."); self.stacked_widget.setCurrentWidget(self.finished_widget); self.main_button_widget.hide()
            self.media_player.stop()
            self.feedback_label.setText(self.tr('feedback_finished')); self.set_feedback_style('finished')
            image_file = self.tr('final_image_file'); pixmap = QPixmap(IMAGE_PATH + image_file)
            if pixmap.isNull(): print(f"Error: Finished image not found or invalid: {IMAGE_PATH + image_file}"); self.finished_image_label.setText(f"{self.tr('feedback_finished')} (Image Error)")
            else:
                label_size = self.finished_image_label.size()
                if label_size.width() == 0 or label_size.height() == 0: label_size = self.finished_image_label.maximumSize()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.finished_image_label.setPixmap(scaled_pixmap)
            self.play_again_button.setText(self.tr('button_play_again'))
            self.create_summary_button.setText(self.tr('button_create_summary'))
            self.exit_button.setText(self.tr('button_exit'))
            self.play_sound('FINISHED')
            self._generate_and_save_report()

    def set_feedback_style(self, status): # Keep as is
        palette = self.feedback_label.palette(); color = QColor('lightgray')
        if status == 'CORRECT_STRONG': color = QColor('lightgreen')
        elif status == 'CORRECT_WEAK': color = QColor('lightyellow')
        elif status == 'INCORRECT_MOVEMENT': color = QColor('lightcoral')
        elif status == 'finished': color = QColor('lightblue')
        elif status == 'error': color = QColor('orangered')
        palette.setColor(QPalette.Window, color); self.feedback_label.setPalette(palette)

    def play_sound(self, sound_key): # Keep as is
        if sound_key in self.sound_effects and self.sound_effects[sound_key]:
            sound_to_play = self.sound_effects[sound_key]
            for effect_playing in self.sound_effects.values():
                 if effect_playing is not None and effect_playing.isPlaying(): effect_playing.stop()
            if sound_to_play.isLoaded(): sound_to_play.play()
            else: print(f" -> Warning: Sound {sound_key} not loaded. Attempting play anyway..."); sound_to_play.play()

    def setup_emg_threading(self):
        """Set up EMG data acquisition and processing in separate threads"""
        # Clean up any existing threads first
        self.cleanup_emg_threads()
        
        try:
            print("Setting up EMG threading...")
            
            # Create threads
            self.daq_thread = QThread()
            self.processing_thread = QThread()
            
            # Create workers - handle missing model gracefully
            model_path = "best_temporal_emg_model.pkl"
            if not os.path.exists(model_path):
                print(f"Model file {model_path} not found.")
            else:
                print(f"Loading EMG model from {model_path}")
                self.processing_worker = EMGProcessingWorker(model_path=model_path)
            
            self.daq_worker = EMGDataAcquisitionWorker(use_mock=True)
            
            # Move workers to threads
            self.daq_worker.moveToThread(self.daq_thread)
            self.processing_worker.moveToThread(self.processing_thread)
            
            # Connect signals for DAQ worker
            self.daq_worker.data_ready.connect(self.processing_worker.process_data_chunk)
            self.daq_worker.status_changed.connect(self.on_daq_status_changed)
            self.daq_worker.error_occurred.connect(self.on_daq_error)
            self.daq_worker.finished.connect(self.on_daq_finished)
            
            # Connect signals for processing worker
            self.processing_worker.new_result.connect(self.handle_emg_result)
            self.processing_worker.movement_detected.connect(self.on_movement_detected)
            self.processing_worker.error_occurred.connect(self.on_processing_error)
            self.processing_worker.finished.connect(self.on_processing_finished)
            
            # Connect thread started signals
            self.daq_thread.started.connect(self.daq_worker.start_acquisition)
            self.processing_thread.started.connect(self.processing_worker.initialize_classifier)
            
            # Connect cleanup signals
            self.daq_worker.finished.connect(self.daq_thread.quit)
            self.processing_worker.finished.connect(self.processing_thread.quit)
            
            # Connect thread finished to deleteLater
            self.daq_thread.finished.connect(self.daq_thread.deleteLater)
            self.processing_thread.finished.connect(self.processing_thread.deleteLater)
            
            # Track when threads are deleted
            self.daq_thread.finished.connect(lambda: setattr(self, 'daq_thread', None))
            self.processing_thread.finished.connect(lambda: setattr(self, 'processing_thread', None))
            
            # Start processing thread first, then DAQ thread after a delay
            self.processing_thread.start()
            QTimer.singleShot(500, self.start_daq_thread_safely)
            
            self.emg_running = True
            print("EMG threading setup completed successfully")
            
            # Update UI
            if hasattr(self, 'emg_status_label'):
                self.emg_status_label.setText("EMG Status: Starting...")
            if hasattr(self, 'emg_start_button'):
                self.emg_start_button.setText("Stop EMG Acquisition")
            
        except Exception as e:
            print(f"Error setting up EMG threading: {e}")
            self.cleanup_emg_threads()
            if hasattr(self, 'emg_status_label'):
                self.emg_status_label.setText(f"EMG Status: Setup failed - {e}")
            raise

    def toggle_emg_acquisition(self):
        """Start or stop EMG acquisition with better error handling"""
        try:
            if self.emg_running and self.daq_thread and self.daq_thread.isRunning():
                # Stop acquisition
                print("Stopping EMG acquisition...")
                self.cleanup_emg_threads()
                self.emg_start_button.setText("Start EMG Acquisition")
                if hasattr(self, 'emg_status_label'):
                    self.emg_status_label.setText("EMG Status: Stopped")
            else:
                # Start acquisition
                print("Starting EMG acquisition...")
                self.setup_emg_threading()
        except Exception as e:
            print(f"Error in toggle_emg_acquisition: {e}")
            self.cleanup_emg_threads()
            self.emg_start_button.setText("Start EMG Acquisition")
            if hasattr(self, 'emg_status_label'):
                self.emg_status_label.setText(f"EMG Status: Error - {e}")
    
    def cleanup_emg_threads(self):
        """Safely cleanup EMG threads"""
        print("Cleaning up EMG threads...")
        
        # Stop workers first
        if hasattr(self, 'daq_worker') and self.daq_worker:
            try:
                self.daq_worker.stop_acquisition()
            except Exception as e:
                print(f"Error stopping DAQ worker: {e}")
        
        if hasattr(self, 'processing_worker') and self.processing_worker:
            try:
                self.processing_worker.stop_processing()
            except Exception as e:
                print(f"Error stopping processing worker: {e}")
        
        # Wait for threads to finish and clean up
        if hasattr(self, 'daq_thread') and self.daq_thread:
            try:
                if self.daq_thread.isRunning():
                    self.daq_thread.quit()
                    if not self.daq_thread.wait(2000):
                        print("Warning: DAQ thread did not finish gracefully")
                        self.daq_thread.terminate()
                        self.daq_thread.wait(1000)
            except Exception as e:
                print(f"Error cleaning up DAQ thread: {e}")
        
        if hasattr(self, 'processing_thread') and self.processing_thread:
            try:
                if self.processing_thread.isRunning():
                    self.processing_thread.quit()
                    if not self.processing_thread.wait(2000):
                        print("Warning: Processing thread did not finish gracefully")
                        self.processing_thread.terminate()
                        self.processing_thread.wait(1000)
            except Exception as e:
                print(f"Error cleaning up processing thread: {e}")
        
        # Reset references
        self.daq_thread = None
        self.processing_thread = None
        self.daq_worker = None
        self.processing_worker = None
        self.emg_running = False
        
        print("EMG threads cleanup completed")

    @Slot()
    def on_daq_finished(self):
        """Handle DAQ worker finished"""
        print("DAQ worker finished")

    @Slot()
    def on_processing_finished(self):
        """Handle processing worker finished"""
        print("Processing worker finished")

    @Slot(str)
    def on_daq_status_changed(self, status):
        """Handle DAQ status changes"""
        print(f"DAQ Status: {status}")
        if hasattr(self, 'emg_status_label'):
            self.emg_status_label.setText(f"EMG Status: {status}")

    @Slot(str)
    def on_daq_error(self, error_msg):
        """Handle DAQ errors"""
        print(f"DAQ Error: {error_msg}")
        if hasattr(self, 'emg_status_label'):
            self.emg_status_label.setText(f"EMG Status: DAQ Error - {error_msg}")

    @Slot(str)
    def on_processing_error(self, error_msg):
        """Handle processing errors"""
        print(f"Processing Error: {error_msg}")
        if hasattr(self, 'emg_status_label'):
            self.emg_status_label.setText(f"EMG Status: Processing Error - {error_msg}")

    @Slot(str, float)
    def on_movement_detected(self, movement_type, confidence):
        """Handle detected movements"""
        print(f"Movement detected: {movement_type} (confidence: {confidence:.2f})")

    def start_daq_thread_safely(self):
        """Safely start the DAQ thread with error handling"""
        try:
            if hasattr(self, 'daq_thread') and self.daq_thread and not self.daq_thread.isRunning():
                self.daq_thread.start()
                print("DAQ thread started successfully")
            else:
                print("DAQ thread already running or not available")
        except Exception as e:
            print(f"Error starting DAQ thread: {e}")
            if hasattr(self, 'emg_status_label'):
                self.emg_status_label.setText(f"EMG Status: DAQ start failed - {e}")

    def start_emg_with_test_file(self):
        """Start EMG with a test file"""
        from PySide6.QtWidgets import QFileDialog
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select EMG Test File", "", 
            "NumPy Files (*.npy);;All Files (*)", 
            options=options
        )
        
        if file_path:
            print(f"Starting EMG with test file: {file_path}")
            # Stop current acquisition if running
            if hasattr(self, 'daq_worker') and self.daq_worker:
                self.cleanup_emg_threads()
            
            # Wait a moment for cleanup
            QTimer.singleShot(1000, lambda: self.setup_emg_with_file(file_path))

    def setup_emg_with_file(self, file_path):
        """Set up EMG system with a specific test file"""
        try:
            # Clean up any existing threads first
            self.cleanup_emg_threads()
            
            # Create threads
            self.daq_thread = QThread()
            self.processing_thread = QThread()
            
            # Create workers with test file
            model_path = "best_temporal_emg_model.pkl"
            if not os.path.exists(model_path):
                print(f"Model file {model_path} not found.")
            else:
                self.processing_worker = EMGProcessingWorker(model_path=model_path)
            
            # Use test file for DAQ
            self.daq_worker = EMGDataAcquisitionWorker(use_mock=True, mock_file=file_path)
            
            # Move workers to threads
            self.daq_worker.moveToThread(self.daq_thread)
            self.processing_worker.moveToThread(self.processing_thread)
            
            # Connect all the signals (same as in setup_emg_threading)
            self.daq_worker.data_ready.connect(self.processing_worker.process_data_chunk)
            self.daq_worker.status_changed.connect(self.on_daq_status_changed)
            self.daq_worker.error_occurred.connect(self.on_daq_error)
            self.daq_worker.finished.connect(self.on_daq_finished)
            
            self.processing_worker.new_result.connect(self.handle_emg_result)
            self.processing_worker.movement_detected.connect(self.on_movement_detected)
            self.processing_worker.error_occurred.connect(self.on_processing_error)
            self.processing_worker.finished.connect(self.on_processing_finished)
            
            self.daq_thread.started.connect(self.daq_worker.start_acquisition)
            self.processing_thread.started.connect(self.processing_worker.initialize_classifier)
            
            self.daq_worker.finished.connect(self.daq_thread.quit)
            self.processing_worker.finished.connect(self.processing_thread.quit)
            
            self.daq_thread.finished.connect(self.daq_thread.deleteLater)
            self.processing_thread.finished.connect(self.processing_thread.deleteLater)
            
            self.daq_thread.finished.connect(lambda: setattr(self, 'daq_thread', None))
            self.processing_thread.finished.connect(lambda: setattr(self, 'processing_thread', None))
            
            # Start threads
            self.processing_thread.start()
            QTimer.singleShot(500, self.start_daq_thread_safely)
            
            self.emg_running = True
            self.emg_start_button.setText("Stop EMG Acquisition")
            self.emg_status_label.setText(f"EMG Status: Using test file - {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"Error setting up EMG with test file: {e}")
            if hasattr(self, 'emg_status_label'):
                self.emg_status_label.setText(f"EMG Status: Test file setup failed - {e}")
        
    def setup_emg_controls(self):
        """Set up EMG data acquisition controls with test file option"""
        # Create a frame for EMG controls
        emg_frame = QFrame()
        emg_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        emg_layout = QVBoxLayout(emg_frame)
        emg_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title label
        title_label = QLabel("EMG Data Acquisition")
        title_label.setStyleSheet("font-weight: bold;")
        emg_layout.addWidget(title_label)
        
        # Start/Stop button
        button_layout = QHBoxLayout()
        self.emg_start_button = QPushButton("Start EMG Acquisition")
        self.emg_start_button.clicked.connect(self.toggle_emg_acquisition)
        button_layout.addWidget(self.emg_start_button)
        
        # Test button
        self.emg_test_button = QPushButton("Test with Recording")
        self.emg_test_button.clicked.connect(self.start_emg_with_test_file)
        button_layout.addWidget(self.emg_test_button)
        
        emg_layout.addLayout(button_layout)
        
        # Status label
        self.emg_status_label = QLabel("EMG Status: Initializing...")
        emg_layout.addWidget(self.emg_status_label)
        
        # Add the frame to the main layout at the bottom
        central_widget = self.centralWidget()
        if central_widget and central_widget.layout():
            central_widget.layout().addWidget(emg_frame)
    
    
    def closeEvent(self, event):
        """Clean shutdown of threads"""
        print("Shutting down EMG threads...")
        
        # Use the cleanup method instead of direct calls
        self.cleanup_emg_threads()
        
        # Close Arduino if connected
        if hasattr(self, 'arduino') and self.arduino and self.arduino.is_open:
            print("Closing serial port.")
            self.arduino.close()
        
        # Call parent closeEvent
        if getattr(self, 'close_for_restart_flag', False):
            print("Accepting close event for restart.")
            event.accept()
        else:
            print("Accepting close event and quitting application.")
            event.accept()
            QApplication.instance().quit()
  

    @Slot(QMediaPlayer.MediaStatus)
    def handle_media_status_changed(self, status: QMediaPlayer.MediaStatus): # Keep as is
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            if 0 <= self.current_step_index < len(self.current_exercise_steps_definition):
                print("Video ended, looping..."); self.media_player.setPosition(0); self.media_player.play()

    @Slot(dict)
    def handle_emg_result(self, result):
        # Update plot data
        if 'plot_data' in result:
            plot_data = result['plot_data']
            
            # Check if plot_data is 2D (multiple channels)
            if isinstance(plot_data, list) and isinstance(plot_data[0], list):
                # It's a 2D array - take channel 0 or average the channels
                import numpy as np
                
                # Option 1: Just use channel 0 (biceps)
                plot_data_1d = np.array(plot_data[0])
                
                # Option 2: Or average both channels
                # plot_data_1d = np.mean(np.array(plot_data), axis=0)
                
            else:
                # It's already 1D, just convert to numpy array
                import numpy as np
                plot_data_1d = np.array(plot_data)
                
            # Now set the 1D data to the plot
            self.emg_curve.setData(plot_data_1d)
            
        # Skip if not in an exercise step
        if not (0 <= self.current_step_index < len(self.current_exercise_steps_definition)):
            return
            
        # Get data from result
        timestamp = result.get('timestamp', time.time())
        movement_type = result.get('movement', 'Rest')
        confidence = result.get('confidence', 0.0)
        intensity = result.get('intensity', 0.0)

        #print(f"Movement: {movement_type}, Conf: {confidence:.2f}, Int: {intensity:.4f}")
        
        # Update movement tracker
        completed_movement = self.movement_tracker.update(movement_type, confidence, timestamp)
        
        # Get expected movement for this step
        step_info = self.current_exercise_steps_definition[self.current_step_index]
        expected_movement = step_info.get('expected_movement', None)
        step_confidence_threshold = step_info.get('feedback_threshold', 0.65)
        
        # Determine status based on movement and step requirements
        if confidence < 0.4:  # Low confidence movements are treated as rest
            status = 'NO_MOVEMENT'
        elif expected_movement and expected_movement != movement_type:
            # Wrong movement type detected
            status = 'INCORRECT_MOVEMENT'
        else:
            # Correct movement type
            if confidence >= step_confidence_threshold:
                status = 'CORRECT_STRONG'
            else:
                status = 'CORRECT_WEAK'
        
        # Track attempts
        if status in self.current_step_attempts:
            self.current_step_attempts[status] += 1
        
        # Update feedback
        status_to_key_map = {
            'NO_MOVEMENT': 'feedback_no_movement',
            'INCORRECT_MOVEMENT': 'feedback_incorrect',
            'CORRECT_WEAK': 'feedback_weak',
            'CORRECT_STRONG': 'feedback_strong'
        }
        feedback_key = status_to_key_map.get(status, 'feedback_initializing')
        
        # Add movement type to feedback if available
        if movement_type != 'Rest' and confidence > 0.5:
            movement_feedback = f" ({movement_type}: {confidence:.2f})"
            self.feedback_label.setText(self.tr(feedback_key) + movement_feedback)
        else:
            self.feedback_label.setText(self.tr(feedback_key))
        
        self.set_feedback_style(status)
        self.play_sound(status)
        
        # Advance on completed correct movement
        if completed_movement and completed_movement['type'] == expected_movement:
            if self.last_successful_status_time == 0:
                self.last_successful_status_time = timestamp
                print(f"Movement completed: {completed_movement['type']} ({completed_movement['duration']:.2f}s)")
                def show_next_step_feedback_and_schedule_advance():
                    self.feedback_label.setText(self.tr('feedback_next_step'))
                    self.play_sound('NEXT_STEP')
                    QTimer.singleShot(ADVANCE_DELAY_MS, lambda: self.advance_step(intensity=intensity))

                QTimer.singleShot(SHORT_DELAY_FOR_STRONG_FEEDBACK_MS, show_next_step_feedback_and_schedule_advance)

    @Slot()
    def advance_step(self, intensity=0.0):
        """Advance to the next step after a successful movement"""
        if self.current_step_start_time is not None and \
        0 <= self.current_step_index < len(self.current_exercise_steps_definition):
            # Record step completion data
            time_taken_for_step = time.time() - self.current_step_start_time
            completed_step_info = self.current_exercise_steps_definition[self.current_step_index]
            
            # Get movement data for this step
            expected_movement = completed_step_info.get('expected_movement', None)
            matching_movements = [m for m in self.movement_tracker.completed_movements 
                                if m['type'] == expected_movement]
            
            # Calculate movement metrics
            movement_count = len(matching_movements)
            avg_duration = 0
            avg_confidence = 0
            
            if matching_movements:
                avg_duration = sum(m['duration'] for m in matching_movements) / movement_count
                avg_confidence = sum(m['confidence'] for m in matching_movements) / movement_count
            
            # Create step report
            step_data = {
                'step_id': completed_step_info['id'],
                'step_name_en': completed_step_info['name_en'], 
                'step_name_pt': completed_step_info['name_pt'],
                'time_taken_seconds': round(time_taken_for_step, 2),
                'incorrect_attempts': self.current_step_attempts.get('INCORRECT_MOVEMENT', 0),
                'weak_attempts': self.current_step_attempts.get('CORRECT_WEAK', 0),
                'no_movement_attempts': self.current_step_attempts.get('NO_MOVEMENT', 0),
                'movement_count': movement_count,
                'avg_movement_duration': round(avg_duration, 2),
                'avg_movement_confidence': round(avg_confidence, 2)
            }
            
            self.session_report_data.append(step_data)
            print(f"Step {self.current_step_index + 1} completed. Data: {step_data}")
        
        # Reset for next step
        self.current_step_start_time = None
        self.last_successful_status_time = 0
        
        # Send robot command if needed
        if self.current_step_index >= 0 and intensity > 0:
            if 0 <= self.current_step_index < len(self.current_exercise_steps_definition):
                step_info = self.current_exercise_steps_definition[self.current_step_index]
                self.send_robot_command(step_info['id'], intensity)
        
        # Clear movement history for the next step
        self.movement_tracker.completed_movements = []
        
        # Advance to next step
        self.current_step_index += 1
        self.load_step()

    @Slot()
    def manual_advance_step(self): # Keep as is
        print("Manual advance requested.")
        if self.current_step_start_time is not None and \
           0 <= self.current_step_index < len(self.current_exercise_steps_definition):
            time_taken_for_step = time.time() - self.current_step_start_time
            current_step_info = self.current_exercise_steps_definition[self.current_step_index]
            step_data = {
                'step_id': current_step_info['id'],
                'step_name_en': current_step_info['name_en'], 'step_name_pt': current_step_info['name_pt'],
                'movement_type_en': current_step_info.get('movement_type_en', 'N/A'),
                'movement_type_pt': current_step_info.get('movement_type_pt', 'N/A'),
                'time_taken_seconds': round(time_taken_for_step, 2),
                'incorrect_attempts': self.current_step_attempts.get('INCORRECT_MOVEMENT', 0),
                'weak_attempts': self.current_step_attempts.get('CORRECT_WEAK', 0),
                'no_movement_attempts': self.current_step_attempts.get('NO_MOVEMENT', 0),
                'manually_advanced': True
            }
            self.session_report_data.append(step_data)
            print(f"Step {self.current_step_index + 1} manually advanced. Data: {step_data}")
        self.current_step_start_time = None
        if 0 <= self.current_step_index < len(self.current_exercise_steps_definition):
            step_info = self.current_exercise_steps_definition[self.current_step_index]
            self.send_robot_command(step_info['id'], 0.1)
        else: print("Cannot send command for manual advance, sequence finished or not started.")
        self.current_step_index += 1
        self.load_step()

    def send_robot_command(self, step_id, intensity): # Keep as isZXC
        if self.arduino and self.arduino.is_open:
            velocity = int(min(max(intensity * 255, 0), 255)); command = f"<{step_id}:{velocity}>\n"
            try: self.arduino.write(command.encode('utf-8')); print(f"Sent to Arduino: {command.strip()}")
            except serial.SerialException as e: print(f"Error writing to Arduino: {e}"); self.arduino_status = "error"; self.update_arduino_status_label()

    @Slot()
    def toggle_language(self): # Keep as is (already calls adjustSize/update)
        current_idx = LANGUAGES.index(self.current_language); next_idx = (current_idx + 1) % len(LANGUAGES)
        self.current_language = LANGUAGES[next_idx]
        if hasattr(QApplication.instance(), 'current_app_language'):
            QApplication.instance().current_app_language = self.current_language
        print(f"MainWindow language changed to: {self.current_language}")
        self.setWindowTitle(self.tr('window_title'))
        self.manual_next_button.setText(self.tr('button_next_manual')); self.manual_next_button.adjustSize(); self.manual_next_button.update()
        self.lang_button.setText(self.tr('button_change_lang')); self.lang_button.adjustSize(); self.lang_button.update()
        self.help_button.setText(self.tr('button_help')); self.help_button.adjustSize(); self.help_button.update()
        self.update_arduino_status_label()
        self.play_again_button.setText(self.tr('button_play_again'))
        self.create_summary_button.setText(self.tr('button_create_summary'))
        self.exit_button.setText(self.tr('button_exit'))
        self.load_step()

    @Slot()
    def restart_exercise(self): # Keep as is
        print("Play Again clicked. Requesting restart.")
        self.close_for_restart_flag = True
        if self.processing_thread and self.processing_thread.isRunning():
            print("Signaling worker to stop for restart...")
            self.worker.stop()
            try:
                self.worker.new_result.disconnect(self.handle_emg_result); self.worker.stopped.disconnect(self.on_worker_stopped)
                self.processing_thread.started.disconnect(self.worker.run); self.worker.stopped.disconnect(self.processing_thread.quit)
                self.worker.stopped.disconnect(self.worker.deleteLater); self.processing_thread.finished.disconnect(self.processing_thread.deleteLater)
            except RuntimeError: pass
            self.processing_thread.quit()
            if not self.processing_thread.wait(500): print("Warning: Worker thread did not stop quickly during restart.")
        self.close()

    @Slot()
    def close_application(self): self.close_for_restart_flag = False; self.close()
    @Slot()
    def show_help(self): QMessageBox.information(self, self.tr('help_message_title'), self.tr('help_message_text'))
    @Slot()
    def on_worker_stopped(self): print("Worker thread has confirmed stopped.")

    def _generate_and_save_report(self):
        """Generates a Markdown report of the session and saves it to a file."""
        print("Generating session report...")
        if not self.session_start_time or not self.session_report_data:
            print("No session data to report.")
            return

        # Base directory for the current patient
        patient_data_dir = os.path.join(OUTPUT_PATH, self.patient_name)
        # Directory for individual session reports
        individual_reports_path = os.path.join(patient_data_dir, "individual_reports")
        
        try:
            os.makedirs(individual_reports_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating patient output directory {individual_reports_path}: {e}")
            return

        timestamp_str = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        # Save individual report in its specific subfolder
        individual_report_filename = os.path.join(individual_reports_path, f"report_{timestamp_str}.md")
        # Path for the "latest" report copy
        latest_report_filename = os.path.join(patient_data_dir, "latest_session_report.md")

        total_duration = datetime.datetime.now() - self.session_start_time
        total_duration_str = str(total_duration).split('.')[0]
        
        md_content = []
        md_content.append(f"# {self.tr('report_title')}")
        md_content.append(f"**{self.tr('report_datetime')}:** {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        md_content.append(f"**{self.tr('report_patient_header')}:** {self.patient_name}") # Use translated "Patient"

        current_sequence_trans_key = ""
        if self.current_sequence_name in EXERCISE_SEQUENCES: # self.current_sequence_name is the internal key
            current_sequence_trans_key = EXERCISE_SEQUENCES[self.current_sequence_name][0]
        
        display_sequence_name = self.tr(current_sequence_trans_key) if current_sequence_trans_key else self.current_sequence_name
        
        md_content.append(f"**{self.tr('report_sequence_name')}:** {display_sequence_name}")
        md_content.append(f"**{self.tr('report_total_duration')}:** {total_duration_str}")
        md_content.append(f"\n## {self.tr('report_step_details_header')}")

        md_content.append(f"| {self.tr('report_table_header_step_num')} | Step ID | Step Name (EN) | {self.tr('report_table_header_name')} |  Movement Type (EN) | {self.tr('report_table_header_movement_type')} | {self.tr('report_table_header_time')} | {self.tr('report_table_header_incorrect')} | {self.tr('report_table_header_weak')} | {self.tr('report_table_header_no_movement')} | {self.tr('report_table_header_manual_advance')} |")
        md_content.append("|---|---|---|---|---|---|---|---|---|---|---|")

        for i, step_data in enumerate(self.session_report_data):
            # Explicitly get the translated name based on self.current_language
            if self.current_language == 'pt':
                step_name_for_report_column = step_data.get('step_name_pt', step_data.get('step_name_en', 'N/A'))
            else: # Default to English
                step_name_for_report_column = step_data.get('step_name_en', 'N/A')
            
            step_id = step_data.get('step_id', 'N/A')
            step_name_en = step_data.get('step_name_en', 'N/A') # Canonical English name
            movement_type_en_val = step_data.get('movement_type_en', 'N/A')
            movement_type_lang_val = step_data.get(f'movement_type_{self.current_language}', movement_type_en_val)
            manually_advanced_str = self.tr('report_value_yes') if step_data.get('manually_advanced', False) else self.tr('report_value_no')
            
            md_content.append(f"| {i+1} | {step_id} | {step_name_en} | {step_name_for_report_column} | {movement_type_en_val} | {movement_type_lang_val} | {step_data['time_taken_seconds']:.2f} | "
                              f"{step_data['incorrect_attempts']} | {step_data['weak_attempts']} | "
                              f"{step_data['no_movement_attempts']} | {manually_advanced_str} |")
        try:
            # Save the individual report
            with open(individual_report_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(md_content))
            print(f"Report saved to: {individual_report_filename}")

            # Copy to "latest_session_report.md"
            shutil.copy2(individual_report_filename, latest_report_filename)
            print(f"Latest report copy saved to: {latest_report_filename}")

        except IOError as e:
            print(f"Error writing report file {individual_report_filename} or copying: {e}")
        except shutil.Error as e:
            print(f"Error copying latest report: {e}")


    @Slot()
    def _create_summary_report(self):
        """Handles the 'Create Summary' button click to generate reports in a separate thread."""
        print(f"Create Summary button clicked for patient: {self.patient_name}")
            # Ask user if they want to attempt PDF generation
        reply = QMessageBox.question(self, self.tr('summary_report_title'), # Dialog Title
                                     "Generate PDF report as well?\n(Requires Pandoc and LaTeX (xelatex) to be installed)", # Question
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, # Buttons
                                     QMessageBox.StandardButton.No) # Default button

        attempt_pdf = (reply == QMessageBox.StandardButton.Yes)

        self.create_summary_button.setEnabled(False)
        self.statusBar().showMessage("Generating summary report, please wait...")

        self.pdf_thread = QThread(self)
        self.pdf_worker = PDFGenerationWorker(
            patient_name=self.patient_name,
            base_output_path=OUTPUT_PATH,
            app_strings_dict=STRINGS,
            app_exercise_sequences_def=EXERCISE_SEQUENCES,
            attempt_pdf=attempt_pdf
        )
        self.pdf_worker.moveToThread(self.pdf_thread)

        self.pdf_thread.started.connect(self.pdf_worker.process_report_generation)
        self.pdf_worker.finished.connect(self.on_pdf_generation_finished)
        self.pdf_worker.error.connect(self.on_pdf_generation_error)

        self.pdf_thread.finished.connect(self.pdf_thread.deleteLater)
        self.pdf_worker.finished.connect(self.pdf_worker.deleteLater)

        self.pdf_thread.start()

    @Slot(str)
    def on_pdf_generation_finished(self, status_message):
        """Called when PDF generation worker is done successfully."""
        print("MainWindow: PDF generation finished signal received.")
        QMessageBox.information(self, self.tr('summary_report_title'), status_message)

# --- Main Execution Block ---
if __name__ == '__main__':
    sys.exit(run_application())