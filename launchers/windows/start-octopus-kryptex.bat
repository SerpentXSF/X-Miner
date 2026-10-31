@echo off
REM X-Miner - Conflux (CFX, Octopus) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: cfx-us.kryptex.network:7027!cfx.kryptex.network:7027
cd /d "%~dp0"

set WALLET=YOUR_KRYPTEX_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=cfx-us.kryptex.network:7027!cfx.kryptex.network:7027

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%%
REM   Octopus is limited by the DAG mix's memory bandwidth (it needs a GPU with 10 GB or more): a memory overclock
REM   helps, e.g. --gpu-mclock 2700 on the RX 9070 XT (+7.5%).
set TUNING=

:loop
x-miner.exe --algorithm octopus --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
