@echo off
REM X-Miner - RETRO BOOMER (TAPE) on rplant.xyz (1% pool fee)
REM Edit WALLET (your tape1... TAPE address) and WORKER below, then double-click this file.
cd /d "%~dp0"

set WALLET=YOUR_TAPE_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=stratum-na.rplant.xyz:7066

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Measured examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%% --gpu-voffset -50   (+3.8%% at the same power)
REM   RTX 3060:   set TUNING=--gpu-mclock 810   (same speed, 15 W less)
set TUNING=

:loop
x-miner.exe --algorithm tape --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
