@echo off
REM X-Miner - Ergo (ERG, Autolykos v2) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: erg-us.kryptex.network:7021!erg.kryptex.network:7021
cd /d "%~dp0"

set WALLET=YOUR_KRYPTEX_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=erg-us.kryptex.network:7021!erg.kryptex.network:7021

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%%
REM   Needs about 8 GB of GPU memory (a 7.6 GB table, rebuilt at every block).
set TUNING=

:loop
x-miner.exe --algorithm autolykos2 --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
