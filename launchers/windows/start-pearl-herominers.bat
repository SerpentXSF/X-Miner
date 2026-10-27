@echo off
REM X-Miner - Pearl (PRL) on HeroMiners (prl1... wallet address)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: us.pearl.herominers.com:1200
cd /d "%~dp0"

set WALLET=YOUR_PRL_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=us.pearl.herominers.com:1200

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Example:
REM set TUNING=--gpu-plimit 70%% --gpu-cclock 1500 --gpu-mclock 7000 --gpu-coffset 100 --gpu-voffset -50 --gpu-fan 70
set TUNING=

:loop
x-miner.exe --algorithm pearlhash --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
