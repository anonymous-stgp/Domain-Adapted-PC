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
echo [1/9] Running matching.py ...
python code\matching.py
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- Table1 (metric comparison: weighted_ks, mean_ks, sinkhorn_wasserstein, marginal_energy) ----
echo [2/9] Running Table1.R (all metrics, K=7, both years) ...
Rscript --vanilla code\Table1.R
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- WD Transfer (TF) group ----
echo [3/9] Running TF group (thinnedSV, ANN, XGBoost) ...
echo   Running Table2_TF_thinned_SV.R ...
Rscript --vanilla code\Table2_TF_thinned_SV.R
if errorlevel 1 goto :error
echo   Running Table2_TF_ANN.py ...
python code\Table2_TF_ANN.py
if errorlevel 1 goto :error
echo   Running Table2_TF_XGBoost.py ...
python code\Table2_TF_XGBoost.py
if errorlevel 1 goto :error
echo   thinned twinGP for TF group comes from Table1.R weighted_ks output (already computed above).
echo Done.
echo.

:: ---- Geographic-neighbor transfer (G) group ----
echo [4/9] Running G group (ANN, thinned twinGP, Random Forest, XGBoost, SVR) ...
echo   Running Table2_G_ANN.py ...
python code\Table2_G_ANN.py
if errorlevel 1 goto :error
echo   Running G_thinned_twinGP.R ...
Rscript --vanilla code\G_thinned_twinGP.R
if errorlevel 1 goto :error
echo   Running Table2_G_rf.R ...
Rscript --vanilla code\Table2_G_rf.R
if errorlevel 1 goto :error
echo   Running Table2_G_XGBoost.R ...
Rscript --vanilla code\Table2_G_XGBoost.R
if errorlevel 1 goto :error
echo   Running Table2_G_SVR.R ...
Rscript --vanilla code\Table2_G_SVR.R
if errorlevel 1 goto :error
echo Done.
echo.

:: ---- Pooled training (P) group ----
echo [5/9] Running P group (ANN, thinned twinGP, XGBoost, GNN, Binning) ...
echo   Running Table2_P_ANN.py ...
python code\Table2_P_ANN.py
if errorlevel 1 goto :error
echo   Running P_thinned_twinGP.R ...
Rscript --vanilla code\P_thinned_twinGP.R
if errorlevel 1 goto :error
echo   Running Table2_P_XGBoost.R ...
Rscript --vanilla code\Table2_P_XGBoost.R
if errorlevel 1 goto :error
echo   Running Table2_P_GNN.py ...
python code\Table2_P_GNN.py
if errorlevel 1 goto :error
echo   Running Table2_P_binning.R ...
Rscript --vanilla code\Table2_P_binning.R
if errorlevel 1 goto :error
echo   Note: STGP and BHM (pooled group) are not reproduced here -- see README.
echo Done.
echo.

:: ---- Figure 5 (K sweep, ensemble vs concat) ----
echo [6/9] Running Figure 5 scripts (K=2..10, twinGP and ANN) ...
echo   Running Figure5_twingp.R ...
Rscript --vanilla code\Figure5_twingp.R
if errorlevel 1 goto :error
echo   Running Figure5_ann.py ...
python code\Figure5_ann.py
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
