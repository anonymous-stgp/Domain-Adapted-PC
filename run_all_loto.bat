@echo off
cd /d "%~dp0"
set "OUTDIR=%cd%\results\intermediate"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo ============================================
echo Running ALL LOTO scripts
echo Output directory: %OUTDIR%
echo ============================================
echo.

:: ---- matching ----
echo [1/12] Running matching.py ...
python code\matching.py
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- Table1 LOTO ----
echo [2/12] Running Table1.R  ...
Rscript --vanilla code\Table1.R  
echo Done.
echo.

:: ---- Table2 TF ----
echo [10/12] Running Table2_TF_thinned_SV.R ...
Rscript --vanilla "code\Table2_TF_thinned_SV.R" 
if errorlevel 1 goto :error
echo Running Table2_TF_ANN.py ...
python code\Table2_TF_ANN.py 
if errorlevel 1 goto :error
echo Running Table2_TF_LGBM.py ...
python code\Table2_TF_LGBM.py
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- Table2 G ----
echo [11/12] Running Table2_G scripts ...
Rscript --vanilla code\Table2_G_rf.R 
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_G_SVR.R 
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_G_XGBoost.R 
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- Table2 P ----
echo [12/12] Running Table2_P scripts ...
Rscript --vanilla code\Table2_P_binning.R
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_P_twinGP.R
if errorlevel 1 goto :error
Rscript --vanilla code\Table2_P_XGBoost.R
if errorlevel 1 goto :error
python code\Table2_P_GNN.py 
if errorlevel 1 goto :error
echo Done.
echo.

echo ============================================
echo ALL LOTO scripts finished successfully.
echo ============================================
pause
exit /b 0

:error
echo.
echo FAILED. Check output above.
pause
exit /b 1
