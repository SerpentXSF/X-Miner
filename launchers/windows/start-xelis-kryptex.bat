@echo off
REM X-Miner - Xelis (XEL, XelisHashV3) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: xel-us.kryptex.network:7019!xel.kryptex.network:7019
cd /d "%~dp0"

set WALLET=YOUR_KRYPTEX_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=xel-us.kryptex.network:7019!xel.kryptex.network:7019

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%%
REM   XelisHashV3 gives each hash a 544 KB scratchpad of random reads: memory latency bound, low power draw.
set TUNING=

:loop
x-miner.exe --algorithm xelishashv3 --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
