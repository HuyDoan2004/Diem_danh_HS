@echo off
echo ====================================
echo   ATTENDANCE SYSTEM - QUICK START
echo ====================================
echo.

:menu
echo Please select an option:
echo.
echo 1. Import students from Excel
echo 2. Collect student data (video recording)
echo 3. Train face recognition model
echo 4. Evaluate model quality
echo 5. Fine-tune YOLO for face detection
echo 6. Run attendance system (with camera)
echo 7. Run dashboard (web interface)
echo 8. Setup database
echo 9. Install dependencies
echo 0. Exit
echo.

set /p choice="Enter your choice: "

if "%choice%"=="1" goto import_excel
if "%choice%"=="2" goto collect
if "%choice%"=="3" goto train
if "%choice%"=="4" goto evaluate
if "%choice%"=="5" goto yolo
if "%choice%"=="6" goto attendance
if "%choice%"=="7" goto dashboard
if "%choice%"=="8" goto setup_db
if "%choice%"=="9" goto install
if "%choice%"=="0" goto end

goto menu

:import_excel
echo.
echo ========================================
echo   IMPORT STUDENTS FROM EXCEL
echo ========================================
echo.
echo This will import student list from Excel file
echo and automatically create folders for each student.
echo.
pause
python src\import_students.py
pause
goto menu

:collect
echo.
echo ========================================
echo   COLLECTING STUDENT VIDEO DATA
echo ========================================
echo.
python src\collect_data.py
pause
goto menu

:train
echo.
echo ========================================
echo   TRAINING FACE RECOGNITION MODEL
echo ========================================
echo.
python src\train_model.py
pause
goto menu

:evaluate
echo.
echo ========================================
echo   EVALUATING MODEL QUALITY
echo ========================================
echo.
echo This will generate comprehensive evaluation reports
echo including confusion matrices, metrics, and charts.
echo.
pause
python src\evaluate_model.py
pause
goto menu

:yolo
echo.
echo ========================================
echo   YOLO FINE-TUNING FOR FACE DETECTION
echo ========================================
echo.
echo This will fine-tune YOLOv8 model on your enrollment data
echo for optimized face detection in classroom environment.
echo.
pause
python src\yolo_finetune.py
pause
goto menu

:attendance
echo.
echo ========================================
echo   RUNNING ATTENDANCE SYSTEM
echo ========================================
echo.
echo Make sure:
echo - Camera is connected (RealSense D435i or Webcam)
echo - Face recognition model is trained
echo - Students are registered in database
echo.
pause
python src\attendance_system.py
pause
goto menu

:dashboard
echo.
echo ========================================
echo   RUNNING WEB DASHBOARD
echo ========================================
echo.
echo Dashboard will open at: http://localhost:8501
echo.
streamlit run src\dashboard.py
pause
goto menu

:setup_db
echo.
echo ========================================
echo   DATABASE SETUP
echo ========================================
echo.
python src\database.py
pause
goto menu

:install
echo.
echo ========================================
echo   INSTALLING DEPENDENCIES
echo ========================================
echo.
pip install -r requirements.txt.txt
echo.
echo Installation complete!
pause
goto menu

:end
echo.
echo Goodbye!
echo.
exit
