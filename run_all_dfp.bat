@echo off
cd /d "%~dp0"
set "OUTDIR=%cd%\results\intermediate"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo ============================================
echo Running ALL DFP scripts
echo Output directory: %OUTDIR%
echo ============================================
echo.

:: ---- matching ----
echo [1/12] Running matching_dfp.py ...
python code\matching_dfp.py
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- DFP(thinned twingp) ----
echo [2/12] Running Table1_dfp.R ...
Rscript --vanilla code\Table1_dfp.R weighted_ks
if errorlevel 1 goto :error

echo Done.
echo.

:: ---- Table2 TF DFP ----
echo [3/12] Running Table2_TF DFP scripts ...
Rscript --vanilla "code\Table2_TF_thinned_SV_dfp.R" 2017
if errorlevel 1 goto :error
Rscript --vanilla "code\Table2_TF_thinned_SV_dfp.R" 2018
if errorlevel 1 goto :error
python code\Table2_TF_ANN_dfp.py 2017
if errorlevel 1 goto :error
python code\Table2_TF_ANN_dfp.py 2018
if errorlevel 1 goto :error
echo Running Table2_TF_LGBM_dfp.py ...
python code\Table2_TF_LGBM_dfp.py
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- Table2 G DFP ----
echo [4/12] Running Table2_G DFP scripts ...
Rscript --vanilla code\Table2_G_rf_dfp.R
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_G_SVR_dfp.R 2017
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_G_SVR_dfp.R 2018
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_G_XGBoost_dfp.R 
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- Table2 P DFP ----
echo [5/12] Running Table2_P DFP scripts ...
Rscript --vanilla code\Table2_P_binning_dfp.R
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_P_twinGP_dfp.R 
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_P_XGBoost_dfp.R 
if errorlevel 1 goto :error
python code\Table2_P_GNN_dfp.py
if errorlevel 1 goto :error
echo Done.
echo.

echo ============================================
echo ALL DFP scripts finished successfully.
echo ============================================
pause
exit /b 0

:error
echo.
echo FAILED. Check output above.
pause
exit /b 1
