import os

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from eddie_helper.make_scripts import run_python_script, run_stage_script
from common_paths import eddie_yiming_data_folder, eddie_yiming_deriv_folder, eddie_yiming_models_folder, eddie_yiming_csv_path, eddie_datastore


def filepath_from_mouse_day_sessions(mouse, day, sessions, path_to_all_filepaths):
    
    all_filepaths = pd.read_csv(path_to_all_filepaths)
    sessions_filepaths = []
    
    for session in sessions:
        session_column = all_filepaths.query(f'mouse == {mouse} & day == {day} & session == "{session}"')
        filepath = session_column['filepath'].values[0]
        sessions_filepaths.append(filepath)

    return sessions_filepaths

def main():

    parser = ArgumentParser()

    parser.add_argument('--mice')
    parser.add_argument('--days')
    parser.add_argument('--sessions')
    parser.add_argument('--bodyparts')
    parser.add_argument('--data_folder', default="")
    parser.add_argument('--deriv_folder', default="")
    parser.add_argument("--models_folder", default=None, help="Folder where you keep your dlc models")

    args = parser.parse_args()
    mice = [int(mouse) for mouse in args.mice.split(",")]
    days = [int(day) for day in args.days.split(",")]
    sessions = args.sessions.split(",")
    bodyparts = args.bodyparts.split(",")
    
    data_folder = parser.parse_args().data_folder
    if len(data_folder) == 0:
        data_folder = eddie_yiming_data_folder
    data_folder = Path(data_folder)

    deriv_folder = parser.parse_args().deriv_folder
    if len(deriv_folder) == 0:
        deriv_folder = eddie_yiming_deriv_folder
    deriv_folder = Path(deriv_folder)

    models_folder = parser.parse_args().models_folder
    if models_folder is None:
        models_folder = eddie_yiming_models_folder
    models_folder = Path(models_folder)
    
    for mouse in mice:
        for day in days:
            for session in sessions:
                for bodypart in bodyparts:
                    mouse_string = f"{mouse:02d}"
                    day_string = f"{day:02d}"
                    
                    recording_paths = filepath_from_mouse_day_sessions(mouse, day, sessions=[session], path_to_all_filepaths=eddie_yiming_csv_path)

                    do_stagein_job = False
                    stagein_dict = {}
                    for recording_path in recording_paths:
                        if "OF" in recording_path:
                            video_name = f'M{mouse_string}_D{day_string}_*_{session}.avi'
                            session_type_folder = data_folder / 'OF'
                        else:
                            video_name = f'M{mouse_string}_D{day_string}_side_capture_{session}.avi'
                            session_type_folder = data_folder / 'VR'    
                        output_path = session_type_folder / video_name

                        if len(list(output_path.parent.glob(video_name))) == 0:
                            stagein_dict[f"{eddie_datastore / recording_path}"] = session_type_folder
                            do_stagein_job = True
                            
                        session_type_folder.mkdir(exist_ok=True)
                        
                    stagein_job_name = f"M{mouse}D{day}{session[:2]}in" 
                    run_python_name = f"M{mouse}D{day}{session[:2]}{bodypart}"
                    stageout_job_name = f"M{mouse}D{day}{session[:2]}out" 

                    stageout_dict = {deriv_folder / f"M{mouse:02d}/D{day:02d}/{session}/dlc_output_{bodypart}": eddie_datastore / "derivatives" / f"M{mouse:02d}/D{day:02d}/{session}/"}

                    uv_directory = os.getcwd()
                    python_arg = f"pose_estimation.py --mice={mouse} --days={day} --sessions={session} --bodyparts={bodypart} --data_folder={data_folder} --deriv_folder={deriv_folder} --models_folder={models_folder}"

                    stagein_dependency = None
                    if do_stagein_job:
                        run_stage_script(stagein_dict, job_name=stagein_job_name)
                        stagein_dependency = stagein_job_name
                    run_python_script(uv_directory, python_arg, cores=8, email="y.zhao@ed.ac.uk", staging=False, hold_jid=stagein_dependency, job_name=run_python_name)
                    run_stage_script(stageout_dict, job_name=stageout_job_name, hold_jid=run_python_name)

if __name__ == "__main__":
    main()
