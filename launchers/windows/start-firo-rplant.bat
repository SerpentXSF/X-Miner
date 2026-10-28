@echo off
REM X-Miner - Firo (FIRO, FiroPow) on rplant.xyz (your FIRO address; not yet live-tested by us)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: na.rplant.xyz:7045
cd /d "%~dp0"

set WALLET=YOUR_FIRO_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=na.rplant.xyz:7045

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%%
REM   FiroPow is limited by memory bandwidth: the memory clock / offset raise it, the core clock does not.
set TUNING=

:loop
x-miner.exe --algorithm firopow --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
