@echo off
REM X-Miner - Ethereum Classic (ETC, Etchash) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: etc-us.kryptex.network:7033!etc.kryptex.network:7033
cd /d "%~dp0"

set WALLET=YOUR_KRYPTEX_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=etc-us.kryptex.network:7033!etc.kryptex.network:7033

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%%
REM   Etchash is limited by memory: raise the memory clock / offset; the core clock barely matters.
set TUNING=

:loop
x-miner.exe --algorithm etchash --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
