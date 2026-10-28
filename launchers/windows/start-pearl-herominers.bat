@echo off
REM X-Miner - Pearl (PRL) on HeroMiners (prl1... wallet address)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: us.pearl.herominers.com:1200
cd /d "%~dp0"

set WALLET=YOUR_PRL_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=us.pearl.herominers.com:1200

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Measured examples:
REM   RX 9070 XT, best TH/W: set TUNING=--gpu-plimit 70%%
REM   RTX 3060, best TH/W:   set TUNING=--gpu-cclock 1500
REM   Pearl on NVIDIA: do not lock the memory clock (--gpu-mclock), it costs 8-28%%.
set TUNING=

:loop
x-miner.exe --algorithm pearlhash --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
