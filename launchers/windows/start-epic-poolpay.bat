@echo off
REM X-Miner - Epic Cash (EPIC, ProgPoW) on epic.pool-pay.com (your epicbox address; the worker is joined with +)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: epic.pool-pay.com:3333
cd /d "%~dp0"

set WALLET=YOUR_EPICBOX_ADDRESS@epicbox.epiccash.com
set WORKER=%COMPUTERNAME%
set POOL=epic.pool-pay.com:3333

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%%
REM   The pool credits every share at a fixed difficulty and answers about 60 shares per second, so X-Miner sends at most 50 per second.
set TUNING=

:loop
x-miner.exe --algorithm progpow-epic --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
