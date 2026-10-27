@echo off
REM X-Miner - Parano1d (NOID) on rplant.xyz (1% pool fee)
REM Edit WALLET (your o1... NOID address) and WORKER below, then double-click this file.
cd /d "%~dp0"

set WALLET=YOUR_NOID_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=na.rplant.xyz:17185

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Example:
REM set TUNING=--gpu-plimit 70%% --gpu-cclock 1500 --gpu-mclock 7000 --gpu-coffset 100 --gpu-voffset -50 --gpu-fan 70
set TUNING=

:loop
x-miner.exe --algorithm noid --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
