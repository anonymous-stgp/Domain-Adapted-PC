@echo off
cd /d "%~dp0"
set "OUTDIR=%cd%\results\intermediate"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo ============================================
echo Running ALL DFP scripts
echo (G and TF groups only -- P group is not needed for DFP)
echo Output directory: %OUTDIR%
echo ============================================
echo.

:: ---- matching (DFP) ----
echo [1/5] Running matching_dfp.py ...
python code\matching_dfp.py
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- thinned twinGP (TF group, DFP) via Table1_dfp.R weighted_ks ----
echo [2/5] Running Table1_dfp.R weighted_ks (thinned twinGP for TF group) ...
Rscript --vanilla code\Table1_dfp.R weighted_ks
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- WD Transfer (TF) group, DFP ----
echo [3/5] Running TF DFP group (thinnedSV, ANN, XGBoost) ...
echo   Running Table2_TF_thinned_SV_dfp.R ...
Rscript --vanilla code\Table2_TF_thinned_SV_dfp.R
if errorlevel 1 goto :error
echo   Running Table2_TF_ANN_dfp.py ...
python code\Table2_TF_ANN_dfp.py
if errorlevel 1 goto :error
echo   Running Table2_TF_XGBoost_dfp.py ...
python code\Table2_TF_XGBoost_dfp.py
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- Geographic-neighbor transfer (G) group, DFP ----
echo [4/5] Running G DFP group (ANN, thinned twinGP, Random Forest, XGBoost, SVR) ...
echo   Running Table2_G_ANN_dfp.py ...
python code\Table2_G_ANN_dfp.py
if errorlevel 1 goto :error
echo   Running G_thinned_twinGP_dfp.R ...
Rscript --vanilla code\G_thinned_twinGP_dfp.R
if errorlevel 1 goto :error
echo   Running Table2_G_rf_dfp.R ...
Rscript --vanilla code\Table2_G_rf_dfp.R
if errorlevel 1 goto :error
echo   Running Table2_G_XGBoost_dfp.R ...
Rscript --vanilla code\Table2_G_XGBoost_dfp.R
if errorlevel 1 goto :error
echo   Running Table2_G_SVR_dfp.R ...
Rscript --vanilla code\Table2_G_SVR_dfp.R
if errorlevel 1 goto :error
echo Done.
echo.

echo ============================================
echo ALL DFP scripts finished successfully.
echo (P group skipped -- not needed for the DFP table)
echo ============================================
pause
exit /b 0

:error
echo.
echo FAILED. Check output above.
pause
exit /b 1
