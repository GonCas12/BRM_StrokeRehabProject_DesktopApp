#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summary Report Generator for Stroke Rehabilitation Assistant - Title-Based Language Detection
------------------------------------------------------------------------------------------
This module reads individual session reports for a patient,
aggregates the data, and generates a summary Markdown report in
both English and Portuguese.
- Improved parsing of individual reports by detecting the report's language
  from its main title, then using language-specific keys for metadata.
"""

import os
import datetime
import re
from collections import defaultdict
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for saving plots to files
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pypandoc # Convert md to pdf
import shutil # To check for executables like xelatex

# Helper function to translate using the main app's STRINGS and current language
def tr(text_key, language, strings_dict, default_text=None):
    """Translates a text key using the provided language and strings dictionary."""
    if default_text is None:
        default_text = f"<{text_key}>"
    return strings_dict.get(language, {}).get(text_key, default_text)


def parse_session_report(filepath, app_strings_dict):
    """
    Parses an individual session Markdown report file.
    Determines the report's original language from its title for metadata parsing.
    Returns a dictionary with extracted data, or None if parsing fails.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                print(f"Error: Report file is empty: {filepath}")
                return None
            content = "".join(lines)
            first_line = lines[0].strip()
    except IOError:
        print(f"Error: Could not read report file: {filepath}")
        return None

    data = {
        'date_time': None, 'patient_name': None, 'sequence_name_reported': None,
        'total_duration_seconds': 0, 'steps': [], 'report_original_language': None
    }
    
    detected_report_lang = None
    # Determine language from the title
    title_en = tr('report_title', 'en', app_strings_dict, "Rehabilitation Session Report")
    title_pt = tr('report_title', 'pt', app_strings_dict, "Relatório da Sessão de Reabilitação")

    if first_line.startswith(f"# {title_en}"):
        detected_report_lang = 'en'
    elif first_line.startswith(f"# {title_pt}"):
        detected_report_lang = 'pt'
    
    if detected_report_lang:
        data['report_original_language'] = detected_report_lang
        print(f"Detected language for {filepath} from title: {detected_report_lang}")
    else:
        print(f"Warning: Could not determine language from title for {filepath}. Using fallback 'en'.")
        detected_report_lang = 'en'

    # Parse metadata using the detected_report_lang
    datetime_key_text = tr('report_datetime', detected_report_lang, app_strings_dict)
    patient_key_text = tr('report_patient_header', detected_report_lang, app_strings_dict)
    sequence_key_text = tr('report_sequence_name', detected_report_lang, app_strings_dict)
    duration_key_text = tr('report_total_duration', detected_report_lang, app_strings_dict)

    datetime_key_re = re.escape(datetime_key_text)
    patient_key_re = re.escape(patient_key_text)
    sequence_key_re = re.escape(sequence_key_text)
    duration_key_re = re.escape(duration_key_text)

    datetime_match = re.search(rf"\*\*{datetime_key_re}:\*\*?\s*(.+)", content, re.IGNORECASE)
    patient_match = re.search(rf"\*\*{patient_key_re}:\*\*?\s*(.+)", content, re.IGNORECASE)
    sequence_match = re.search(rf"\*\*{sequence_key_re}:\*\*?\s*(.+)", content, re.IGNORECASE)
    duration_match = re.search(rf"\*\*{duration_key_re}:\*\*?\s*(\d+):(\d+):(\d+)", content, re.IGNORECASE)

    if datetime_match:
        try:
            data['date_time'] = datetime.datetime.strptime(datetime_match.group(1).strip(), '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"Warning: Could not parse date_time from {filepath} using key '{datetime_key_text}'")
    if patient_match:
        data['patient_name'] = patient_match.group(1).strip()
    if sequence_match:
        data['sequence_name_reported'] = sequence_match.group(1).strip()
    if duration_match:
        hours, minutes, seconds = map(int, duration_match.groups())
        data['total_duration_seconds'] = hours * 3600 + minutes * 60 + seconds

    # Table parsing: look for the localized table header row
    hdr_cols = [
        tr('report_table_header_step_num', detected_report_lang, app_strings_dict),
        'Step ID',
        'Step Name (EN)',
        tr('report_table_header_name', detected_report_lang, app_strings_dict),
        'Movement Type (EN)',
        tr('report_table_header_movement_type', detected_report_lang, app_strings_dict),
        tr('report_table_header_time', detected_report_lang, app_strings_dict),
        tr('report_table_header_incorrect', detected_report_lang, app_strings_dict),
        tr('report_table_header_weak', detected_report_lang, app_strings_dict),
        tr('report_table_header_no_movement', detected_report_lang, app_strings_dict),
        tr('report_table_header_manual_advance', detected_report_lang, app_strings_dict),
    ]

    hdr_regex = r"\|\s*" + r"\s*\|\s*".join(re.escape(h) for h in hdr_cols) + r"\s*\|"
    table_header_pattern = hdr_regex

    table_match = re.search(table_header_pattern, content, re.IGNORECASE)
    if table_match:
        table_content = content[table_match.end():] # Allows the search to be taken only in the part after the header
        step_pattern = re.compile(
            r"\|\s*\d+\s*\|"                      # Step #
            r"\s*([^|]*?)\s*\|"                   # Step ID
            r"\s*([^|]*?)\s*\|"                   # Step Name (EN)
            r"\s*([^|])*?\s*\|"                   # Translated Name (ignored)
            r"\s*([^|]*?)\s*\|"                   # Movement Type (EN)
            r"\s*([^|])*?\s*\|"                   # Translated Movement Type (ignored) 
            r"\s*([\d\.]+)\s*\|"                  # Time Taken
            r"\s*(\d+)\s*\|"                      # Incorrect
            r"\s*(\d+)\s*\|"                      # Weak
            r"\s*(\d+)\s*\|"                      # No Movement
            r"\s*(.*?)\s*\|"                      # Manual Advance
        )
        for line in table_content.splitlines():
            match = step_pattern.match(line.strip())
            if match:
                step_id_str = match.group(1).strip()
                step_name_en = match.group(2).strip()
                mov_type_en = match.group(4).strip()
                time_taken = float(match.group(6))
                incorrect = int(match.group(7))
                weak = int(match.group(8))
                no_movement = int(match.group(9))
                manual_advance_str = match.group(10).strip()
                lang_for_yes_no = data['report_original_language'] or 'en'
                manually_advanced = (manual_advance_str.lower() == 
                                    tr('report_value_yes', lang_for_yes_no, app_strings_dict).lower())
                data['steps'].append({
                    'id': step_id_str if step_id_str != "N/A" else None,
                    'name_en': step_name_en if step_name_en else "Unknown Step",
                    'movement_type_en': mov_type_en if mov_type_en != "N/A" else "Unknown Type",
                    'time_taken': time_taken,
                    'incorrect': incorrect,
                    'weak': weak,
                    'no_movement': no_movement,
                    'manually_advanced': manually_advanced
                })
    else:
        print(f"Warning: Could not find step details table header in {filepath}.")

    if not data['date_time'] and not data['steps']:
        print(f"No usable data parsed from {filepath}")
        return None
    return data


def generate_trainings_per_day_graph(session_dates, patient_name, output_image_path, language, strings_dict):
    if not session_dates:
        return None
    sessions_by_day = defaultdict(int)
    for dt in session_dates:
        sessions_by_day[dt.date()] += 1
    sorted_days = sorted(sessions_by_day.keys())
    counts = [sessions_by_day[day] for day in sorted_days]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(sorted_days, counts, width=0.8)
    ax.set_xlabel(tr('graph_xlabel_date', language, strings_dict, "Date"))
    ax.set_ylabel(tr('graph_ylabel_sessions', language, strings_dict, "Number of Sessions"))
    ax.set_title(
        tr('graph_title_sessions_per_day', language, strings_dict, "Training Sessions per Day")
        + f" - {patient_name}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=max(1, len(sorted_days) // 7 if len(sorted_days) > 7 else 1))
    )
    fig.autofmt_xdate()
    plt.tight_layout()
    graph_filename = os.path.join(
        output_image_path,
        f"sessions_per_day_{patient_name}_{language}.png"
    )
    try:
        plt.savefig(graph_filename)
        plt.close(fig)
        print(f"Graph saved: {graph_filename}")
        return graph_filename
    except Exception as e:
        print(f"Error saving graph {graph_filename}: {e}")
        plt.close(fig)
        return None

# Not used
def generate_trainings_per_minute_graph(session_datetimes, patient_name, output_image_path, language, strings_dict):
    """Generates a bar graph of training sessions per minute."""
    if not session_datetimes:
        return None

    sessions_by_minute = defaultdict(int)
    for dt in session_datetimes:
        # Aggregate by minute: clear seconds and microseconds
        minute_timestamp = dt.replace(second=0, microsecond=0)
        sessions_by_minute[minute_timestamp] += 1

    if not sessions_by_minute:
        return None

    sorted_minutes = sorted(sessions_by_minute.keys())
    counts = [sessions_by_minute[minute] for minute in sorted_minutes]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(sorted_minutes, counts, width=0.0005)

    ax.set_xlabel(tr('graph_xlabel_datetime_minute', language, strings_dict, "Date and Time (Minute)"))
    ax.set_ylabel(tr('graph_ylabel_sessions', language, strings_dict, "Number of Sessions"))
    ax.set_title(
        tr('graph_title_sessions_per_minute', language, strings_dict, "Training Sessions per Minute")
        + f" - {patient_name}"
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # Adjust locator for minutes; e.g., every 15 minutes if many, or every minute if few.
    num_distinct_minutes = len(sorted_minutes)
    if num_distinct_minutes > 60: # If more than an hour of distinct minutes
        interval = max(1, num_distinct_minutes // 12) # Aim for around 12 ticks
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
    elif num_distinct_minutes > 10:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5))) # Every 5 mins
    else:
        ax.xaxis.set_major_locator(mdates.MinuteLocator()) # Every minute

    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    graph_filename = os.path.join(
        output_image_path,
        f"sessions_per_minute_{patient_name}_{language}.png"
    )
    try:
        plt.savefig(graph_filename)
        plt.close(fig)
        print(f"Graph saved: {graph_filename}")
        return graph_filename
    except Exception as e:
        print(f"Error saving graph {graph_filename}: {e}")
        plt.close(fig)
        return None

# Global caches, keyed by step_id (string) and step_name_en (string) respectively
ALL_STEPS_DEFINITIONS_CACHE_BY_ID = {}
ALL_STEPS_DEFINITIONS_CACHE_BY_NAME_EN = {}

def _ensure_all_steps_caches_populated(app_exercise_sequences_def):
    global ALL_STEPS_DEFINITIONS_CACHE_BY_ID, ALL_STEPS_DEFINITIONS_CACHE_BY_NAME_EN
    # Populate only if the primary cache (by ID) is empty
    if not ALL_STEPS_DEFINITIONS_CACHE_BY_ID:
        temp_cache_by_id = {}
        temp_cache_by_name_en = {}
        processed_ids = set()

        for _, sequence_data in app_exercise_sequences_def.items():
            # sequence_data is a tuple: (trans_key, steps_list)
            if isinstance(sequence_data, tuple) and len(sequence_data) == 2:
                _, steps_list = sequence_data
                if isinstance(steps_list, list):
                    for step_info in steps_list:
                        if not isinstance(step_info, dict): continue

                        step_id_str = str(step_info.get('id'))
                        step_name_en = step_info.get('name_en')

                        if step_id_str and step_id_str not in processed_ids:
                            temp_cache_by_id[step_id_str] = step_info
                            processed_ids.add(step_id_str)
                        
                        if step_name_en and step_name_en not in temp_cache_by_name_en:
                            # Only add to name_en cache if name is not already a key 
                            temp_cache_by_name_en[step_name_en] = step_info
                            
        ALL_STEPS_DEFINITIONS_CACHE_BY_ID = temp_cache_by_id
        ALL_STEPS_DEFINITIONS_CACHE_BY_NAME_EN = temp_cache_by_name_en
        print(f"CACHE_BY_ID populated with {len(ALL_STEPS_DEFINITIONS_CACHE_BY_ID)} entries.")
        print(f"CACHE_BY_NAME_EN populated with {len(ALL_STEPS_DEFINITIONS_CACHE_BY_NAME_EN)} entries.")


def check_pdf_prerequisites():
    """Checks for Pandoc and a LaTeX engine (xelatex)."""
    pandoc_ok = False
    xelatex_ok = False
    try:
        pypandoc.get_pandoc_version()
        pandoc_ok = True
        print("Pandoc found.")
    except OSError:
        print("Pandoc not found. PDF generation will be skipped.")

    if shutil.which("xelatex"):
        xelatex_ok = True
        print("xelatex (LaTeX engine) found.")
    else:
        print("xelatex (LaTeX engine) not found. PDF generation will be skipped.")
    
    return pandoc_ok and xelatex_ok

def generate_summary_report_for_patient(patient_name, base_output_path, app_strings_dict, app_exercise_sequences_def, attempt_pdf_generation=False):
    print(f"Generating summary reports for patient: {patient_name}...")

    _ensure_all_steps_caches_populated(app_exercise_sequences_def)

    # Main directory for the patient
    patient_main_data_dir = os.path.join(base_output_path, patient_name)
    
    # Individual reports in a subfolder
    individual_reports_dir = os.path.join(patient_main_data_dir, "individual_reports")
    
    # Summary images will be in the patient's main data directory
    summary_images_path = os.path.join(patient_main_data_dir, "summary_images")

    if not os.path.isdir(individual_reports_dir):
        return f"No session data found for {patient_name}."
    os.makedirs(summary_images_path, exist_ok=True)

    report_files = sorted([
        f for f in os.listdir(individual_reports_dir)
        if f.startswith("report_") and f.endswith(".md")
    ])
    if not report_files:
        return f"No individual reports found for {patient_name}."

    all_parsed_sessions = []
    for report_file in report_files:
        filepath = os.path.join(individual_reports_dir, report_file)
        parsed_data = parse_session_report(filepath, app_strings_dict)
        if parsed_data and parsed_data['date_time']:
            all_parsed_sessions.append(parsed_data)
        else:
            print(f"Warning: Could not reliably parse or missing essential data in {filepath}.")

    if not all_parsed_sessions:
        return f"Could not parse any valid data from reports for {patient_name}."

    all_parsed_sessions.sort(key=lambda s: s['date_time'])
    status_messages = []

    # Movement Type translation map
    movement_type_translations = {} # Key: English movement type, Value: {'en': ..., 'pt': ...}
    for step_def in ALL_STEPS_DEFINITIONS_CACHE_BY_ID.values():
        en_type = step_def.get('movement_type_en')
        if en_type and en_type not in movement_type_translations:
            movement_type_translations[en_type] = {
                'en': en_type,
                'pt': step_def.get('movement_type_pt', en_type)
            }

    # Loop to generate report in each language ('en','pt')
    for report_lang in app_strings_dict.keys():
        print(f"Generating {report_lang} summary...")

        num_total_sessions = len(all_parsed_sessions)
        total_training_duration_seconds = sum(s['total_duration_seconds'] for s in all_parsed_sessions)
        mean_session_duration_seconds = (
            total_training_duration_seconds / num_total_sessions
            if num_total_sessions > 0 else 0
        )
        sessions_per_sequence_type = defaultdict(lambda: {
            'count': 0, 'total_duration': 0,
            'steps_data': defaultdict(
                lambda: {'time': [], 'incorrect': 0, 'weak': 0,
                         'no_movement': 0, 'manual': 0,
                         'total_completed':0, 'perfect_first_try':0}
            )
        })
        session_dates = [s['date_time'] for s in all_parsed_sessions]
        total_steps = total_perfect = total_no_incorrect = 0
        overall_error = defaultdict(lambda: {'errors':0, 'count':0, 'name_en':''})

        performance_by_movement_type = defaultdict(lambda: {
            'total_time': 0, 'total_steps': 0, 'incorrect': 0, 'weak': 0,
            'no_movement': 0, 'manual': 0, 'perfect_first_try': 0
        })

        for sess in all_parsed_sessions:
            # identify sequence key by matching any translation
            seq_key = None
            for ik, (trans_key, _) in app_exercise_sequences_def.items():
                if sess['sequence_name_reported'] in (
                    tr(trans_key, 'en', app_strings_dict),
                    tr(trans_key, 'pt', app_strings_dict)
                ):
                    seq_key = ik; break
            if not seq_key:
                seq_key = sess['sequence_name_reported']

            agg = sessions_per_sequence_type[seq_key]
            agg['count'] += 1
            agg['total_duration'] += sess['total_duration_seconds']
            for step in sess['steps']:
                total_steps += 1
                errors = step['incorrect'] + step['weak'] + step['no_movement']
                if not step['manually_advanced'] and errors == 0:
                    total_perfect += 1
                if step['incorrect'] == 0:
                    total_no_incorrect += 1
                key = step['id'] or step['name_en']
                sd = agg['steps_data'][key]
                sd['time'].append(step['time_taken'])
                sd['incorrect'] += step['incorrect']
                sd['weak'] += step['weak']
                sd['no_movement'] += step['no_movement']
                sd['manual'] += int(step['manually_advanced'])
                sd['total_completed'] += 1
                if not step['manually_advanced'] and errors == 0:
                    sd['perfect_first_try'] +=1
                oe = overall_error[key]
                oe['errors'] += errors
                oe['count'] += 1
                oe['name_en'] = step['name_en']
                oe['id'] = step['id']

                # Agregate by Movement Type
                mov_type_en = step.get('movement_type_en', 'Unknown Type')
                mt_data = performance_by_movement_type[mov_type_en]
                mt_data['total_time'] += step['time_taken']
                mt_data['total_steps'] += 1
                mt_data['incorrect'] += step['incorrect']
                mt_data['weak'] += step['weak']
                mt_data['no_movement'] += step['no_movement']
                mt_data['manual'] += 1 if step['manually_advanced'] else 0
                if not step['manually_advanced'] and errors == 0:
                    mt_data['perfect_first_try'] += 1


        success_pct = (total_perfect/total_steps*100) if total_steps>0 else 0
        no_error_pct = (total_no_incorrect/total_steps*100) if total_steps>0 else 0

        # find most problematic step
        most_problematic_step_display_name = tr('report_value_na', report_lang, app_strings_dict)
        errs = {k:v for k,v in overall_error.items() if v['errors']>0}
        if errs:
            most_problematic_step_key = max(errs, key=lambda k: errs[k]['errors'])
            most_step_id = errs[most_problematic_step_key].get('id')
            most_step = errs[most_problematic_step_key]['name_en']

            step_definition = None
            if most_step_id:
                step_definition = ALL_STEPS_DEFINITIONS_CACHE_BY_ID.get(most_step_id)
            if not step_definition:
                step_definition = ALL_STEPS_DEFINITIONS_CACHE_BY_NAME_EN.get(most_step)
            
            if step_definition:
                most_problematic_step_display_name = step_definition.get(f'name_{report_lang}', most_step)
            else:
                most_problematic_step_display_name = most_step

            
        # most problematic sequence
        seq_errors = {k: sum(d['incorrect']+d['weak']+d['no_movement']
                        for d in agg['steps_data'].values())
                      for k,agg in sessions_per_sequence_type.items()}
        most_seq = tr('report_value_na', report_lang, app_strings_dict)
        if seq_errors:
            ms = max(seq_errors, key=lambda k: seq_errors[k])
            trans = app_exercise_sequences_def.get(ms,(ms,None))[0]
            most_seq = tr(trans, report_lang, app_strings_dict, ms)

        graph_fn_daily = generate_trainings_per_day_graph(
            session_dates, patient_name, summary_images_path, report_lang, app_strings_dict
        )
        # graph_fn_minutely = generate_trainings_per_minute_graph(
        #     session_dates, patient_name, summary_images_path, report_lang, app_strings_dict
        # )

        # build Markdown
        md = [
            f"# {tr('summary_report_title', report_lang, app_strings_dict)} - {patient_name}",
            f"*{tr('report_generated_on', report_lang, app_strings_dict)}:"
             + f" {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "", f"## {tr('summary_overview_header', report_lang, app_strings_dict)}",
            f"- **{tr('summary_total_sessions', report_lang, app_strings_dict)}:** {num_total_sessions}",
            f"- **{tr('summary_total_train_time', report_lang, app_strings_dict)}:** "
             + f"{str(datetime.timedelta(seconds=int(total_training_duration_seconds)))}",
            f"- **{tr('summary_mean_session_time', report_lang, app_strings_dict)}:** "
             + f"{str(datetime.timedelta(seconds=int(mean_session_duration_seconds)))}",
            f"- **{tr('summary_overall_step_success_rate', report_lang, app_strings_dict)}:** "
             + f"{success_pct:.1f}%",
            f"- **{tr('summary_overall_step_no_incorrect_rate', report_lang, app_strings_dict)}:** "
             + f"{no_error_pct:.1f}%",
            f"- **{tr('summary_most_problematic_step', report_lang, app_strings_dict)}:** {most_problematic_step_display_name}",
            f"- **{tr('summary_most_problematic_sequence', report_lang, app_strings_dict)}:** {most_seq}"
        ]
        if graph_fn_daily:
            absolute_graph_path_daily = os.path.abspath(graph_fn_daily).replace(os.sep, "/")
            md.append(f"\n### {tr('graph_title_sessions_per_day', report_lang, app_strings_dict)}")
            md.append(f"![{tr('graph_title_sessions_per_day', report_lang, app_strings_dict)}]({absolute_graph_path_daily})")
        
        # if graph_fn_minutely:
        #     absolute_graph_path_minutely = os.path.abspath(graph_fn_minutely).replace(os.sep, "/")
        #     md.append(f"\n### {tr('graph_title_sessions_per_minute', report_lang, app_strings_dict, 'Training Sessions per Minute')}")
        #     md.append(f"![{tr('graph_title_sessions_per_minute', report_lang, app_strings_dict, 'Training Sessions per Minute')}]({absolute_graph_path_minutely})")

        # Performance by Movement Type
        md.append(f"\n## {tr('summary_movement_type_header', report_lang, app_strings_dict, 'Performance by Movement Type')}")
        if performance_by_movement_type:
            md.append(f"| {tr('report_table_header_movement_type', report_lang, app_strings_dict, 'Movement Type')} | {tr('summary_avg_time_per_step', report_lang, app_strings_dict)} | {tr('summary_total_incorrect', report_lang, app_strings_dict)} | {tr('summary_total_weak', report_lang, app_strings_dict)} | {tr('summary_total_no_movement', report_lang, app_strings_dict)} | {tr('summary_total_manual_advance', report_lang, app_strings_dict)} | {tr('summary_step_perfection_rate', report_lang, app_strings_dict)} |")
            md.append( "|---|---|---|---|---|---|---|" )
            sorted_mov_types = sorted(performance_by_movement_type.keys())
            for mov_type_en_key in sorted_mov_types:
                mt_data = performance_by_movement_type[mov_type_en_key]
                
                translations_for_current_mov_type = movement_type_translations.get(mov_type_en_key)
                mov_type_display = translations_for_current_mov_type.get(report_lang, mov_type_en_key)

                avg_time = (mt_data['total_time'] / mt_data['total_steps']) if mt_data['total_steps'] > 0 else 0
                perfection_rate = (mt_data['perfect_first_try'] / mt_data['total_steps'] * 100) if mt_data['total_steps'] > 0 else 0
                md.append(f"| {mov_type_display} | {avg_time:.2f} | {mt_data['incorrect']} | {mt_data['weak']} | {mt_data['no_movement']} | {mt_data['manual']} | {perfection_rate:.1f}% |")
        else:
            md.append(f"{tr('report_value_na', report_lang, app_strings_dict, 'N/A')}")

        md.append(f"\n## {tr('summary_sequence_performance_header', report_lang, app_strings_dict)}")
        for seq_internal_key, data_dict in sessions_per_sequence_type.items():
            trans_key_for_seq_name = app_exercise_sequences_def.get(seq_internal_key, (seq_internal_key,None))[0]
            display_seq_name = tr(trans_key_for_seq_name, report_lang, app_strings_dict, seq_internal_key)
            md.append(f"\n### {display_seq_name}")
            md.append(f"- **{tr('summary_times_performed', report_lang, app_strings_dict)}:** {data_dict['count']}")
            mean_duration_this_seq = data_dict['total_duration'] / data_dict['count'] if data_dict['count'] > 0 else 0
            md.append(f"- **{tr('summary_mean_duration_this_sequence', report_lang, app_strings_dict)}:** {str(datetime.timedelta(seconds=int(mean_duration_this_seq)))}")
            md.append(f"\n**{tr('report_step_details_header', report_lang, app_strings_dict)} ({display_seq_name}):**")
            md.append(f"\n| {tr('report_table_header_name', report_lang, app_strings_dict)} | {tr('report_table_header_movement_type', report_lang, app_strings_dict)} | {tr('summary_avg_time_per_step', report_lang, app_strings_dict)} | {tr('summary_total_incorrect', report_lang, app_strings_dict)} | {tr('summary_total_weak', report_lang, app_strings_dict)} | {tr('summary_total_no_movement', report_lang, app_strings_dict)} | {tr('summary_total_manual_advance', report_lang, app_strings_dict)} | {tr('summary_step_perfection_rate', report_lang, app_strings_dict)} |")
            md.append( "|---|---|---|---|---|---|---|---|" )
            current_seq_step_defs = []
            for ik, (_, steps_list) in app_exercise_sequences_def.items():
                if ik == seq_internal_key: current_seq_step_defs = steps_list; break
            for step_def_info in current_seq_step_defs:
                step_id_from_def = str(step_def_info['id']); step_name_en_from_def = step_def_info['name_en']
                step_key_for_agg = step_id_from_def if step_id_from_def in data_dict['steps_data'] else step_name_en_from_def
                step_agg_data = data_dict['steps_data'].get(step_key_for_agg)
                movement_type_for_step = step_def_info.get(f'movement_type_{report_lang}', step_def_info.get('movement_type_en', tr('report_value_na', report_lang, app_strings_dict)))
                if step_agg_data:
                    step_display_name = step_def_info.get(f'name_{report_lang}', step_name_en_from_def)
                    avg_time = sum(step_agg_data['time']) / len(step_agg_data['time']) if step_agg_data['time'] else 0
                    perfection_rate = (step_agg_data['perfect_first_try'] / step_agg_data['total_completed'] * 100) if step_agg_data['total_completed'] > 0 else 0
                    md.append(f"| {step_display_name} | {movement_type_for_step} | {avg_time:.2f} | {step_agg_data['incorrect']} | {step_agg_data['weak']} | {step_agg_data['no_movement']} | {step_agg_data['manual']} | {perfection_rate:.1f}% |")
                else:
                    step_display_name = step_def_info.get(f'name_{report_lang}', step_name_en_from_def)
                    md.append(f"| {step_display_name} | {movement_type_for_step} | {tr('report_value_na', report_lang, app_strings_dict)} | {tr('report_value_na', report_lang, app_strings_dict)} | {tr('report_value_na', report_lang, app_strings_dict)} | {tr('report_value_na', report_lang, app_strings_dict)} | {tr('report_value_na', report_lang, app_strings_dict)} | {tr('report_value_na', report_lang, app_strings_dict)} |")


        out_fn = os.path.join(patient_main_data_dir,
                              f"summary_report_{patient_name}_{report_lang}.md")
        with open(out_fn, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        status_messages.append(f"Summary ({report_lang}) saved: {os.path.basename(out_fn)}")

        if attempt_pdf_generation:
            pdf_output_filename = out_fn.replace(".md", ".pdf")
            pdf_success, pdf_message = md_to_pdf_pypandoc(out_fn, pdf_output_filename)
            status_messages.append(pdf_message)
        else:
            status_messages.append(f"PDF generation ({report_lang}) skipped by user choice.")

    return "\n".join(status_messages) if status_messages else "Summary generation complete."

def md_to_pdf_pypandoc(md_path, pdf_path):
    if not check_pdf_prerequisites():
        return False, "PDF generation skipped: Pandoc or LaTeX engine missing."

    # Ensure image paths use forward slashes:        
    pypandoc.convert_file(
        md_path,
        to="pdf",
        format="markdown+pipe_tables",
        outputfile=pdf_path,
        extra_args=["--pdf-engine=xelatex",
                    "--resource-path=.",
                    "-V", "geometry:margin=2cm"]
    )
    # os.remove(temp_md)
    print(f"Generated PDF via pypandoc: {pdf_path}")
    return True, f"PDF generated: {os.path.basename(pdf_path)}"
