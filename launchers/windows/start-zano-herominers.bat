@echo off
REM X-Miner - Zano (ZANO, ProgPowZ) on HeroMiners (your Zano address)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: us.zano.herominers.com:1110!de.zano.herominers.com:1110
cd /d "%~dp0"

set WALLET=YOUR_ZANO_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=us.zano.herominers.com:1110!de.zano.herominers.com:1110

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%%
REM   RX 9070 XT: 40 MH/s at the -30% power limit; ProgPowZ is compute and memory heavy.
set TUNING=

:loop
x-miner.exe --algorithm progpowz --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
