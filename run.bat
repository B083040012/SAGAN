@echo off
SETLOCAL EnableDelayedExpansion

@REM change the conda environment here
@REM vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
SET conda-env=stdn_keras
CALL activate %conda-env%

:MENU
cls

echo *********************************************
echo ************SAGAN Running Script*************
echo ************weather_only dataset*************

CHOICE /C:12 /M "Select Dataset (1: region, 2: station)"
IF %ERRORLEVEL% == 1 SET dataset=region
IF %ERRORLEVEL% == 2 SET dataset=station

CHOICE /C:1234 /M "Select Start Phase (1: train supernet, 2: seaching, 3: retrain, 4: evaluate)"
set start_phase=%ERRORLEVEL%

CHOICE /C:1234 /M "Select End Phase (1: train supernet, 2: seaching, 3: retrain, 4: evaluate)"
set end_phase=%ERRORLEVEL%

set /P running_times="Enter running times:"

set /P project_name="Enter project_name:"

CHOICE /C:01 /M "Debug Mode ? (False / True)"
IF %ERRORLEVEL% == 1 SET debug=False
IF %ERRORLEVEL% == 2 SET debug=True

echo *********************************************
echo dataset: %dataset%
echo start_phase: %start_phase%
echo end_phase: %end_phase%
echo running_times: %running_times%
echo project_name: %project_name%
echo debug_mode: %debug%
echo *********************************************
CHOICE /C:yn /M "start running project ?"
IF %ERRORLEVEL% == 2 GOTO MENU

python agent.py --dataset=%dataset% --project_name=%project_name% --start_phase=%start_phase% --end_phase=%end_phase% --running_times=%running_times% --debug_mode=%debug%

pause